"""
Модуль рассуждений CogniFlex - внутренний диалог и многоуровневое мышление
Интегрирует все компоненты системы для глубокого анализа перед ответом пользователю
"""

import os
import sys
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger("cogniflex.reasoning")


class ReasoningPhase(Enum):
    """Фазы процесса рассуждения"""
    INITIAL_ANALYSIS = "initial_analysis"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONTRADICTION_CHECK = "contradiction_check"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_GRAPH_QUERY = "knowledge_graph_query"
    ETHICS_CHECK = "ethics_check"
    SYNTHESIS = "synthesis"
    REFLECTION = "reflection"
    FINAL_ANSWER = "final_answer"


@dataclass
class ReasoningStep:
    """Шаг внутреннего рассуждения"""
    phase: ReasoningPhase
    query: str
    result: Any
    timestamp: float
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class InternalDialogue:
    """Внутренний диалог системы с собой"""
    query_id: str
    original_query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    contradictions_found: List[Dict] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0


class ReasoningEngine:
    """
    Движок рассуждений CogniFlex - реализует внутренний диалог системы
    
    Перед ответом пользователю система проводит многоуровневый анализ:
    1. Извлечение контекста из памяти (граф знаний, горячее окно)
    2. Проверка на противоречия
    3. Поиск актуальной информации (веб-поиск при необходимости)
    4. Этическая проверка
    5. Синтез ответа через внутренний диалог
    6. Рефлексия и самокоррекция
    """
    
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
        self.reasoning_history: List[InternalDialogue] = []
        self.max_history_size = self.config.get('max_history', 100)
        
        # Настройки фаз рассуждения
        self.enabled_phases = {
            ReasoningPhase.INITIAL_ANALYSIS: True,
            ReasoningPhase.MEMORY_RETRIEVAL: True,
            ReasoningPhase.CONTRADICTION_CHECK: True,
            ReasoningPhase.WEB_SEARCH: self.config.get('enable_web_search', True),
            ReasoningPhase.KNOWLEDGE_GRAPH_QUERY: True,
            ReasoningPhase.ETHICS_CHECK: True,
            ReasoningPhase.SYNTHESIS: True,
            ReasoningPhase.REFLECTION: True,
            ReasoningPhase.FINAL_ANSWER: True
        }
        
        # Порог уверенности для разных действий
        self.thresholds = {
            'min_confidence_for_answer': 0.6,
            'web_search_threshold': 0.4,  # При низкой уверенности - искать в вебе
            'reflection_threshold': 0.7,    # При высокой уверенности - рефлексия
            'max_reasoning_time': 30.0     # Максимальное время рассуждения
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
            # Фаза 1: Начальный анализ
            if self.enabled_phases[ReasoningPhase.INITIAL_ANALYSIS]:
                self._initial_analysis(dialogue, query, context)
            
            # Фаза 2: Извлечение из памяти
            if self.enabled_phases[ReasoningPhase.MEMORY_RETRIEVAL]:
                self._memory_retrieval(dialogue)
            
            # Фаза 3: Проверка на противоречия
            if self.enabled_phases[ReasoningPhase.CONTRADICTION_CHECK]:
                self._check_contradictions(dialogue)
            
            # Фаза 4: Веб-поиск (если нужно)
            if self.enabled_phases[ReasoningPhase.WEB_SEARCH]:
                self._web_search_if_needed(dialogue)
            
            context = {}
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
            
            # Фаза 5: Запрос к графу знаний
            if self.enabled_phases[ReasoningPhase.KNOWLEDGE_GRAPH_QUERY]:
                self._query_knowledge_graph(dialogue)
            
            # Фаза 6: Этическая проверка
            if self.enabled_phases[ReasoningPhase.ETHICS_CHECK]:
                self._ethics_check(dialogue)
            
            # Фаза 7: Синтез через внутренний диалог
            if self.enabled_phases[ReasoningPhase.SYNTHESIS]:
                self._synthesize_answer(dialogue)
            
            # Фаза 8: Рефлексия
            if self.enabled_phases[ReasoningPhase.REFLECTION]:
                self._reflection(dialogue)
            
            # Фаза 9: Финальный ответ
            result = self._finalize_answer(dialogue, start_time)
            
            # Сохраняем в историю
            self._add_to_history(dialogue)
            
            # Автоматически пополняем граф памяти
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
            # Очищаем активный диалог
            if query_id in self.active_dialogues:
                del self.active_dialogues[query_id]
    
    def _initial_analysis(self, dialogue: InternalDialogue, query: str, context: Optional[Dict]):
        """Начальный анализ запроса"""
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
        """Классифицирует тип запроса"""
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
        """Оценивает сложность запроса"""
        words = query.split()
        length = len(words)
        
        if length <= 3:
            return 'simple'
        elif length <= 8:
            return 'medium'
        else:
            return 'complex'
    
    def _extract_entities(self, text: str) -> List[str]:
        """Извлекает сущности из текста"""
        # Простое извлечение существительных
        import re
        # Ищем слова с заглавной буквы (имена собственные) и длинные слова
        entities = []
        
        # Имена собственные
        proper_nouns = re.findall(r'\b[А-Я][а-я]+\b', text)
        entities.extend(proper_nouns)
        
        # Длинные слова (возможно термины)
        long_words = re.findall(r'\b[а-яА-Я]{7,}\b', text)
        entities.extend([w for w in long_words if w.lower() not in 
                        ['сегодня', 'завтра', 'вчера', 'сейчас', 'потому', 'поэтому']])
        
        return list(set(entities))
    
    def _extract_intent(self, query: str) -> str:
        """Извлекает намерение пользователя"""
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
        """Оценивает срочность запроса"""
        urgent_words = ['срочно', 'быстро', 'немедленно', 'сейчас', 'asap']
        if any(w in query.lower() for w in urgent_words):
            return 'high'
        return 'normal'
    
    def _memory_retrieval(self, dialogue: InternalDialogue):
        """Извлечение релевантной информации из памяти"""
        logger.debug("Извлечение информации из памяти...")
        
        memories = []
        
        if self.brain:
            # Получаем данные из горячего окна
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
            
            # Получаем данные из графа знаний
            if hasattr(self.brain, 'knowledge_graph'):
                try:
                    entities = dialogue.steps[0].result.get('entities', [])
                    for entity in entities[:3]:  # Ограничиваем количество
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
            
            # Получаем историю диалога
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
        """Проверка на противоречия в знаниях"""
        logger.debug("Проверка на противоречия...")
        
        contradictions = []
        
        if self.brain and hasattr(self.brain, 'contradiction_manager'):
            try:
                # Проверяем противоречия для извлеченных знаний
                memory_step = self._get_step_by_phase(dialogue, ReasoningPhase.MEMORY_RETRIEVAL)
                if memory_step and memory_step.result:
                    for memory in memory_step.result:
                        if isinstance(memory.get('data'), dict):
                            # Проверяем на противоречия
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
        """Выполняет веб-поиск при недостатке информации"""
        logger.debug("Оценка необходимости веб-поиска...")
        
        # Оцениваем уверенность на основе предыдущих шагов
        memory_step = self._get_step_by_phase(dialogue, ReasoningPhase.MEMORY_RETRIEVAL)
        confidence = memory_step.confidence if memory_step else 0.0
        
        search_results = []
        
        # Если уверенность низкая - ищем в вебе
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
        """Запрос к графу знаний для получения связей с использованием MemoryGraphML"""
        logger.debug("Запрос к графу знаний через MemoryGraphML...")
        
        knowledge = []
        graph_context = {}
        
        # Пробуем использовать MemoryGraphML для получения фрактального контекста
        if self.brain and hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
            try:
                # Получаем многоуровневый фрактальный контекст
                for level in range(self.brain.memory_graph_ml.fractal_levels):
                    level_context = self.brain.memory_graph_ml.get_fractal_context(
                        dialogue.original_query, 
                        level=level
                    )
                    if level_context:
                        graph_context[f'level_{level}'] = level_context
                        # Добавляем связанные концепты
                        related = level_context.get('related_concepts', [])
                        patterns = level_context.get('patterns', [])
                        knowledge.extend([f"[L{level}] {c}" for c in related[:10]])
                        knowledge.extend([f"[Pattern] {p.get('pattern_id', 'unknown')}" for p in patterns[:5]])
                
                logger.debug(f"Получен фрактальный контекст из {len(graph_context)} уровней")
            except Exception as e:
                logger.debug(f"MemoryGraphML недоступен: {e}")
        
        # Fallback на прямой доступ к knowledge_graph
        if not knowledge and self.brain and hasattr(self.brain, 'knowledge_graph'):
            try:
                entities = dialogue.steps[0].result.get('entities', []) if dialogue.steps else []
                
                for entity in entities:
                    # Получаем связанные концепты
                    related = self.brain.knowledge_graph.find_path_between_concepts(
                        entity, dialogue.original_query
                    )
                    if related:
                        knowledge.append(related)
                    
                    # Получаем факты о сущности
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
    
    def _ethics_check(self, dialogue: InternalDialogue):
        """Проверка на соответствие этическим принципам"""
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
    
    def _gather_context(self, dialogue: InternalDialogue) -> Dict[str, Any]:
        """Сбор контекста из всех шагов рассуждения с учетом MemoryGraphML"""
        context = {
            'query_type': '',
            'entities': [],
            'memory': [],
            'contradictions': [],
            'web_search': [],
            'knowledge_graph': [],
            'graph_context': {},
            'ethics': {},
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
        
        # Дополнительно получаем контекст из MemoryGraphML если доступен
        if self.brain and hasattr(self.brain, 'memory_graph_ml') and self.brain.memory_graph_ml:
            try:
                ml_context = self.brain.memory_graph_ml.get_context_for_query(dialogue.original_query)
                if ml_context:
                    context['ml_entities'] = ml_context.get('entities', [])
                    context['ml_related'] = ml_context.get('related_concepts', [])
            except Exception as e:
                logger.debug(f"Ошибка получения ML контекста: {e}")
        
        return context

    def _synthesize_answer(self, dialogue: InternalDialogue):
        """Синтез ответа через внутренний диалог"""
        logger.debug("Синтез ответа через внутренний диалог...")
        
        # Собираем всю информацию из предыдущих шагов
        context = self._gather_context(dialogue)
        
        # Формируем внутренние вопросы
        internal_questions = self._generate_internal_questions(dialogue, context)
        
        insights = []
        
        # Проводим внутренний диалог
        for question in internal_questions:
            answer = self._ask_internal(question, context, dialogue)
            insights.append({
                'question': question,
                'answer': answer,
                'timestamp': time.time()
            })
        
        dialogue.insights = [i['answer'] for i in insights]
        
        step = ReasoningStep(
            phase=ReasoningPhase.SYNTHESIS,
            query=dialogue.original_query,
            result={'insights': insights, 'context': context},
            timestamp=time.time(),
            confidence=self._calculate_synthesis_confidence(insights),
            metadata={'insight_count': len(insights)}
        )
        dialogue.steps.append(step)
        
        logger.debug(f"Синтез завершен: {len(insights)} инсайтов")
    
    def _generate_internal_questions(self, dialogue: InternalDialogue, context: Dict) -> List[str]:
        """Генерация внутренних вопросов на основе контекста"""
        questions = []
        query_type = context.get('query_type', 'general')
        
        # Базовые вопросы для всех типов
        questions.extend([
            f"Какие ключевые аспекты запроса '{dialogue.original_query}' нужно учесть?",
            "Какая информация из памяти наиболее релевантна?",
            "Есть ли пробелы в знаниях, которые нужно заполнить?"
        ])
        
        # Специфичные вопросы по типу
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
        
        # Если есть противоречия
        if context.get('contradictions'):
            questions.append("Как разрешить обнаруженные противоречия?")
        
        # Если мало знаний
        if not context.get('memories') and not context.get('web_search_results'):
            questions.append("Как ответить при недостатке информации?")
        
        return questions[:5]  # Ограничиваем количество
    
    def _ask_internal(self, question: str, context: Dict, dialogue: InternalDialogue) -> str:
        """Задает внутренний вопрос и получает ответ"""
        # Используем генератор для внутреннего ответа
        if self.brain and hasattr(self.brain, 'generation_coordinator'):
            try:
                # Формируем промпт для внутреннего диалога
                prompt = f"""Вопрос для самоанализа: {question}
Контекст: {context.get('original_query', '')}
Ответь кратко и по существу:"""
                
                response = self.brain.generation_coordinator.generate_response(
                    prompt, max_tokens=50
                )
                if response and response.strip():
                    return response
            except Exception as e:
                logger.debug(f"Ошибка внутреннего вопроса: {e}")
        
        # Fallback: пробуем напрямую через brain если generation_coordinator не доступен
        if self.brain:
            try:
                # Пробуем использовать ml_unit напрямую
                if hasattr(self.brain, 'ml_unit') and self.brain.ml_unit:
                    ml_unit = self.brain.ml_unit
                    if hasattr(ml_unit, 'generate_response'):
                        response = ml_unit.generate_response(question)
                        if response and isinstance(response, dict):
                            return response.get('text', response.get('response', ''))
                        elif isinstance(response, str):
                            return response
                # Пробуем model_manager напрямую
                if hasattr(self.brain, 'model_manager') and self.brain.model_manager:
                    mm = self.brain.model_manager
                    if hasattr(mm, 'generate'):
                        response = mm.generate(question)
                        if response:
                            return response
            except Exception as e:
                logger.debug(f"Fallback generation failed: {e}")
        
        # Fallback - возвращаем пустую строку, чтобы не засорять ответ
        return ""
    
    def _calculate_synthesis_confidence(self, insights: List[Dict]) -> float:
        """Вычисляет уверенность на основе инсайтов"""
        if not insights:
            return 0.3
        
        # Базовая уверенность
        base_confidence = 0.5
        
        # Увеличиваем за каждый значимый инсайт
        for insight in insights:
            answer = insight.get('answer', '')
            if len(answer) > 20:  # Непустой ответ
                base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _reflection(self, dialogue: InternalDialogue):
        """Рефлексия - самокоррекция ответа"""
        logger.debug("Фаза рефлексии...")
        
        synthesis_step = self._get_step_by_phase(dialogue, ReasoningPhase.SYNTHESIS)
        if not synthesis_step:
            return
        
        insights = synthesis_step.result.get('insights', [])
        
        # Проверяем качество инсайтов
        reflection_notes = []
        
        for insight in insights:
            answer = insight.get('answer', '')
            # Проверяем на бессвязность
            if len(answer) < 10 or self._is_gibberish(answer):
                reflection_notes.append(f"Инсайт слабый: {insight.get('question', '')}")
        
        # Если много проблем - помечаем для перегенерации
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
        """Проверяет текст на бессвязность"""
        # Простая проверка
        words = text.split()
        if not words:
            return True
        
        # Проверяем среднюю длину слов
        avg_len = sum(len(w) for w in words) / len(words)
        if avg_len > 20:  # Слишком длинные слова
            return True
        
        # Проверяем на повторы
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Слишком много повторов
            return True
        
        return False
    
    def _direct_generate(self, query: str) -> str:
        """
        Прямая генерация ответа без использования внутреннего диалога.
        Используется как fallback когда внутренний диалог не дал результатов.
        """
        # Пробуем через generation_coordinator
        if self.brain and hasattr(self.brain, 'generation_coordinator') and self.brain.generation_coordinator:
            try:
                response = self.brain.generation_coordinator.generate_response(
                    query, max_tokens=150
                )
                if response and response.strip():
                    return response
            except Exception as e:
                logger.debug(f"Ошибка прямой генерации через coordinator: {e}")
        
        # Пробуем через ml_unit
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
        
        # Пробуем через model_manager
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
        
        # Last resort - простой ответ
        return "Я получил ваш запрос, но не могу сейчас сгенерировать полный ответ."
    
    def _finalize_answer(self, dialogue: InternalDialogue, start_time: float) -> Dict:
        """Финализация ответа"""
        logger.debug("Финализация ответа...")
        
        # Собираем все инсайты в финальный ответ
        synthesis_step = self._get_step_by_phase(dialogue, ReasoningPhase.SYNTHESIS)
        
        if synthesis_step and synthesis_step.result:
            insights = synthesis_step.result.get('insights', [])
            
            # Формируем ответ из инсайтов - фильтруем пустые и короткие ответы
            if insights:
                # Фильтруем: убираем пустые ответы и ответы короче 10 символов
                answers = [i.get('answer', '') for i in insights 
                          if i.get('answer') and len(i.get('answer', '')) > 10]
                if answers:
                    final_answer = ' '.join(answers)
                else:
                    # Пробуем прямой вызов генерации если внутренний диалог не дал результатов
                    final_answer = self._direct_generate(dialogue.original_query)
            else:
                # Пробуем прямой вызов генерации
                final_answer = self._direct_generate(dialogue.original_query)
        else:
            # Пробуем прямой вызов генерации
            final_answer = self._direct_generate(dialogue.original_query)
        
        # Вычисляем общую уверенность
        confidence = self._calculate_overall_confidence(dialogue)
        
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
    
    def _calculate_overall_confidence(self, dialogue: InternalDialogue) -> float:
        """Вычисляет общую уверенность ответа"""
        confidences = [s.confidence for s in dialogue.steps]
        if not confidences:
            return 0.3
        
        # Взвешенное среднее
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
    
    def _update_memory_graph(self, dialogue: InternalDialogue):
        """Автоматическое пополнение графа памяти на основе рассуждения"""
        logger.debug("Обновление графа памяти...")
        
        if not self.brain:
            return
        
        try:
            # Добавляем узел для запроса
            if hasattr(self.brain, 'knowledge_graph'):
                self.brain.knowledge_graph.add_concept(
                    id=f"query_{dialogue.query_id}",
                    name=dialogue.original_query,
                    description=dialogue.final_answer[:200],
                    strength=dialogue.confidence,
                    domain="user_queries"
                )
                
                # Добавляем связи для сущностей
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
                
                # Добавляем инсайты как узлы
                for i, insight in enumerate(dialogue.insights[:3]):
                    if insight and len(insight) > 10:
                        self.brain.knowledge_graph.add_concept(
                            id=f"insight_{dialogue.query_id}_{i}",
                            name=f"Инсайт {i+1}",
                            description=insight[:200],
                            strength=0.6,
                            domain="insights"
                        )
                
                logger.debug(f"Граф памяти обновлен для {dialogue.query_id}")
        except Exception as e:
            logger.debug(f"Ошибка обновления графа памяти: {e}")
    
    def _add_to_history(self, dialogue: InternalDialogue):
        """Добавляет диалог в историю"""
        self.reasoning_history.append(dialogue)
        
        # Ограничиваем размер истории
        if len(self.reasoning_history) > self.max_history_size:
            self.reasoning_history = self.reasoning_history[-self.max_history_size:]
    
    def _get_step_by_phase(self, dialogue: InternalDialogue, phase: ReasoningPhase) -> Optional[ReasoningStep]:
        """Получает шаг по фазе"""
        for step in dialogue.steps:
            if step.phase == phase:
                return step
        return None
    
    def get_reasoning_stats(self) -> Dict:
        """Возвращает статистику рассуждений"""
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

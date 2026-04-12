"""
Self-Reasoning Engine - главный движок рассуждения ЕВА
Цикл: Generate → Analyze → Clarify → Repeat until confidence >= 0.75
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List

from .reasoning_types import (
    ReasoningStep,
    ReasoningResult,
    AnalysisResult,
    ReasoningPhase
)
from .confidence_scorer import (
    calculate_overall_confidence,
    should_terminate,
    CONFIDENCE_THRESHOLD,
    get_adaptive_weights,
    get_adaptive_threshold,
    calculate_adaptive_confidence
)
from .clarification_generator import ClarificationGenerator
from .sre_context import (
    _build_contextual_query,
    _get_wikipedia_context,
    _determine_query_type,
    _get_generation_params,
    _generate_with_qwen,
    _generate_simple_response,
    _get_knowledge_response,
)
from .sre_quality import (
    check_quality,
    _sanitize_response,
    _clean_filler_start,
    _remove_looping_blocks,
    _check_relevance,
)
from .sre_feedback import *
from .sre_recursive import (
    _recursive_process_query,
    _check_semantic_stability,
    _recursive_reasoning_step,
    _is_complex_query,
    decompose_query,
    retrieve_similar_reasoning,
    build_recursive_context,
    _synthesize_recursive_results,
    _linear_process_query,
    _init_retriever,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
DEFAULT_MAX_NEW_TOKENS = 2048
MAX_RECURSION_DEPTH = 3


class SelfReasoningEngine:
    """
    Движок самостоятельного рассуждения с поддержкой рекурсии
    Использует ЕДИНСТВЕННЫЙ экземпляр Qwen (singleton) для генерации
    """

    LOGICAL_FACTORS = {
        'ethics': {
            'name': 'Этика',
            'weight': 0.2,
            'description': 'Соответствие этическим нормам'
        },
        'knowledge': {
            'name': 'Знания',
            'weight': 0.25,
            'description': 'Фактическая точность информации'
        },
        'contradiction': {
            'name': 'Противоречия',
            'weight': 0.2,
            'description': 'Отсутствие внутренних противоречий'
        },
        'context': {
            'name': 'Контекст',
            'weight': 0.15,
            'description': 'Учёт контекста запроса'
        },
        'logic': {
            'name': 'Логика',
            'weight': 0.2,
            'description': 'Логическая согласованность'
        }
    }

    def __init__(self, brain, config: Optional[Dict[str, Any]] = None, two_model_pipeline=None):
        self.brain = brain
        self.config = config or {}
        self.two_model_pipeline = two_model_pipeline

        self.max_iterations = self.config.get('max_iterations', MAX_ITERATIONS)
        self.confidence_threshold = self.config.get('confidence_threshold', CONFIDENCE_THRESHOLD)
        self.max_new_tokens = self.config.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
        self.max_recursion_depth = self.config.get('max_recursion_depth', MAX_RECURSION_DEPTH)

        self.clarification_gen = ClarificationGenerator()

        self.fractal_storage = getattr(brain, 'fractal_storage', None) if brain else None

        if self.fractal_storage is None:
            try:
                from eva_ai.reasoning.fractal_ml import FractalStorage
                storage_path = './cache/fractal_reasoning'
                if brain and hasattr(brain, 'cache_dir'):
                    import os
                    storage_path = os.path.join(brain.cache_dir, 'fractal_reasoning')
                self.fractal_storage = FractalStorage(storage_dir=storage_path)
                logger.info(f"FractalStorage создан: {storage_path}")
            except Exception as e:
                logger.warning(f"Не удалось создать FractalStorage: {e}")
                self.fractal_storage = None

        self.fractal_embedder = None
        self.fractal_retriever = None
        self._init_fractal_components()

        self._qwen_cached = None
        self._qwen_init_lock = threading.Lock()

        self.total_queries = 0
        self.total_iterations = 0
        self.recursive_calls = 0

        logger.info(f"SelfReasoningEngine инициализирован: max_iterations={self.max_iterations}, recursion_depth={self.max_recursion_depth}")

    def _init_fractal_components(self):
        """Инициализация FractalRetriever и FractalEmbedder"""
        try:
            from eva_ai.reasoning.fractal_ml.fractal_embedder import FractalEmbedder
            from eva_ai.reasoning.fractal_ml.fractal_retriever import FractalRetriever

            self.fractal_embedder = FractalEmbedder(use_sentence_transformers=False)

            if self.fractal_embedder and self.fractal_storage:
                self.fractal_retriever = FractalRetriever(
                    storage=self.fractal_storage,
                    embedder=self.fractal_embedder
                )
                logger.info("FractalEmbedder и FractalRetriever инициализированы")
            else:
                logger.info("FractalEmbedder инициализирован (retriever будет создан позже)")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать fractal компоненты: {e}")
            self.fractal_embedder = None
            self.fractal_retriever = None

    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        start_time = time.time()
        self.total_queries += 1

        conversation_history = []
        if user_context and 'conversation_history' in user_context:
            conversation_history = user_context['conversation_history']
            logger.info(f"Получена история диалогов: {len(conversation_history)} сообщений")

        enhanced_query = self._build_contextual_query(query, conversation_history)

        logger.debug("Enhanced query built, length = %d", len(enhanced_query))

        pipeline = self.two_model_pipeline
        logger.debug("SRE: checking two_model_pipeline, self=%s", self.two_model_pipeline is not None)
        if pipeline is None and hasattr(self.brain, 'two_model_pipeline'):
            pipeline = self.brain.two_model_pipeline
            logger.debug("SRE: fallback to brain.two_model_pipeline, found=%s", pipeline is not None)
        if pipeline:
            logger.debug("SRE: pipeline found via %s", 'self.two_model_pipeline' if self.two_model_pipeline else 'brain.two_model_pipeline')
        logger.info(f"Проверка Two-Model Pipeline: brain={self.brain is not None}, pipeline={pipeline is not None}")

        if pipeline:
            logger.info("Two-Model Pipeline найден, запускаем генерацию...")
            try:
                logger.info("Используем Two-Model Pipeline для генерации...")

                query_type = self._determine_query_type(query)
                logger.info(f"Тип запроса определён: {query_type}")

                gen_params = self._get_generation_params(query_type)
                logger.info(f"Параметры генерации: {gen_params}")

                pipeline_result = pipeline.process_query(
                    query=query,
                    gen_params=gen_params
                )

                model_a_response = pipeline_result.get('model_a_result', {}).get('natural_response', '')
                model_b_response = pipeline_result.get('model_b_result', {}).get('natural_response', '')
                pipeline_steps = pipeline_result.get('reasoning_steps', [])

                reasoning_steps = []

                reasoning_steps.append({
                    'step': 1,
                    'phase': 'query_analysis',
                    'thought': f'Анализ запроса: {query_type}',
                    'confidence': 0.9,
                    'model': 'System',
                    'action': 'Определение типа запроса'
                })

                if pipeline_steps:
                    for ps in pipeline_steps:
                        if ps.get('phase') == 'model_a_generation':
                            reasoning_steps.append({
                                'step': 2,
                                'phase': 'model_a_generation',
                                'thought': f'Модель А (логика): {model_a_response[:150]}...' if len(model_a_response) > 150 else f'Модель А (логика): {model_a_response}',
                                'confidence': ps.get('confidence', 0.7),
                                'model': 'Model A (Logic)',
                                'action': 'Извлечение фактов',
                                'input': query,
                                'output': model_a_response
                            })

                quality_a = self._check_response_quality(model_a_response, query)
                reasoning_steps.append({
                    'step': 3,
                    'phase': 'quality_check_a',
                    'thought': f'Качество Model A: score={quality_a["score"]:.2f}, gibberish={quality_a["is_gibberish"]}',
                    'confidence': quality_a['score'],
                    'model': 'Quality Checker',
                    'action': 'Проверка качества ответа Model A'
                })

                if pipeline_steps:
                    for ps in pipeline_steps:
                        if ps.get('phase') == 'model_b_generation':
                            reasoning_steps.append({
                                'step': 4,
                                'phase': 'model_b_generation',
                                'thought': f'Модель B (концепции): {model_b_response[:150]}...' if len(model_b_response) > 150 else f'Модель B (концепции): {model_b_response}',
                                'confidence': ps.get('confidence', 0.7),
                                'model': 'Model B (Concept)',
                                'action': 'Расширение концепций',
                                'input': f'Факты: {model_a_response}',
                                'output': model_b_response
                            })

                quality_b = self._check_response_quality(model_b_response, query)
                reasoning_steps.append({
                    'step': 5,
                    'phase': 'quality_check_b',
                    'thought': f'Качество Model B: score={quality_b["score"]:.2f}, gibberish={quality_b["is_gibberish"]}',
                    'confidence': quality_b['score'],
                    'model': 'Quality Checker',
                    'action': 'Проверка качества ответа Model B'
                })

                relevance = self._check_relevance(model_b_response, query)
                reasoning_steps.append({
                    'step': 6,
                    'phase': 'relevance_check',
                    'thought': f'Релевантность: {relevance["score"]:.2f}',
                    'confidence': relevance['score'],
                    'model': 'Relevance Checker',
                    'action': 'Проверка соответствия запросу'
                })

                final_response = model_b_response
                final_confidence = quality_b['score']
                decision_reason = "Запрошен подробный ответ"

                if query_type == 'кратко':
                    if quality_a['score'] >= 0.5 and not quality_a['is_gibberish']:
                        final_response = model_a_response
                        final_confidence = quality_a['score']
                        decision_reason = "Запрошен краткий ответ - используем Model A"
                    else:
                        decision_reason = "Model A низкого качества - используем Model B"

                reasoning_steps.append({
                    'step': 7,
                    'phase': 'final_decision',
                    'thought': decision_reason,
                    'confidence': final_confidence,
                    'model': 'System',
                    'action': 'Выбор финального ответа'
                })

                logger.info(f"Two-Model Pipeline завершён: {decision_reason}")

                return {
                    "response": final_response,
                    "text": final_response,
                    "status": "ok",
                    "confidence": final_confidence,
                    "reasoning_steps": reasoning_steps,
                    "model_a_response": model_a_response,
                    "model_b_response": model_b_response,
                    "query_type": query_type,
                    "source": "self_reasoning_engine",
                    "processing_time": time.time() - start_time
                }

            except Exception as e:
                logger.error(f"Ошибка Two-Model Pipeline в SelfReasoningEngine: {e}", exc_info=True)
                logger.warning("Падение Two-Model Pipeline - пробуем fallback")
        else:
            logger.error("Two-Model Pipeline НЕДОСТУПЕН - нет резервного метода генерации")
            logger.error(f"self.two_model_pipeline: {self.two_model_pipeline is not None}")
            if self.brain:
                logger.error(f"brain.two_model_pipeline: {hasattr(self.brain, 'two_model_pipeline')}")
            return {
                "response": "Ошибка: Two-Model Pipeline недоступен.",
                "text": "Ошибка: Two-Model Pipeline недоступен.",
                "status": "error",
                "confidence": 0.0,
                "error_type": "pipeline_unavailable",
                "error_detail": "Two-Model Pipeline не инициализирован",
                "source": "self_reasoning_engine",
                "processing_time": time.time() - start_time,
                "conversation_history_used": len(conversation_history) > 0
            }

        if self.fractal_retriever and self._is_complex_query(query):
            logger.info("Сложный запрос - используем рекурсивный reasoning")
            result = self._recursive_process_query(query, user_context, depth=0)
            result["processing_time"] = time.time() - start_time
            return result

        result = ReasoningResult(
            final_response="",
            confidence=0.0,
            iterations=0,
            query=query
        )

        current_query = query
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            result.iterations = iteration
            self.total_iterations += 1

            logger.info(f"Итерация {iteration}/{self.max_iterations}")

            step = ReasoningStep(
                phase=ReasoningPhase.GENERATION.value,
                thought=f"Генерирую ответ на: {current_query[:30]}...",
                confidence=0.0
            )

            response = self._generate_with_qwen(current_query)
            result.steps.append(step)

            if not response or response.startswith("Ошибка"):
                logger.warning(f"Ошибка генерации: {response}")
                result.final_response = response or "Ошибка генерации"
                break

            analysis = self._analyze_response(current_query, response)

            adaptive_weights = get_adaptive_weights(iteration)
            current_threshold = get_adaptive_threshold(iteration)

            logger.info(f"Адаптивные веса итерации {iteration}: ethics={adaptive_weights['ethics']:.2f}, "
                       f"contradiction={adaptive_weights['contradiction']:.2f}, knowledge={adaptive_weights['knowledge']:.2f}")
            logger.info(f"Адаптивный порог итерации {iteration}: {current_threshold:.2f}")

            confidence = calculate_adaptive_confidence(
                ethics_result=analysis.ethics_result,
                contradiction_result=analysis.contradiction_result,
                knowledge_result=analysis.knowledge_result,
                query=current_query,
                iteration=iteration
            )

            step = ReasoningStep(
                phase=ReasoningPhase.FINAL_SYNTHESIS.value,
                thought=f"Анализ завершён (итерация {iteration}). Уверенность: {confidence:.2f}, порог: {current_threshold:.2f}",
                confidence=confidence
            )
            result.steps.append(step)

            result.confidence = confidence

            try:
                factors_result = self._analyze_logical_factors(current_query, response, user_context)

                factor_summary = []
                details = factors_result.get('details', {})
                if details and isinstance(details, dict):
                    for factor_name, factor_data in details.items():
                        if not isinstance(factor_data, dict):
                            continue
                        factor_info = self.LOGICAL_FACTORS.get(factor_name, {})
                        fname = factor_info.get('name', factor_name)
                        fscore = factor_data.get('score', 0)
                        factor_summary.append(f"{fname}: {fscore:.2f}")
                        logger.info(f"Фактор {fname}: {fscore:.2f}")

                step = ReasoningStep(
                    phase="logical_analysis",
                    thought=f"Анализ факторов: {', '.join(factor_summary)}",
                    confidence=factors_result.get('overall', {}).get('score', confidence)
                )
                result.steps.append(step)

                if self._should_use_alternative_branch(factors_result):
                    logger.info("Условие выполнено: используем альтернативную ветвь рассуждения")

                    alternatives = self._find_alternative_reasoning_branches(current_query, response, factors_result)

                    if alternatives:
                        for alt in alternatives:
                            logger.info(f"Альтернатива: {alt['factor_name']} (приоритет: {alt['priority']:.2f})")

                        for alt in alternatives:
                            try:
                                alt_response = self._generate_with_qwen(alt['alternative'])
                                alt['generated_response'] = alt_response
                            except Exception as e:
                                logger.debug(f"Ошибка генерации альтернативы: {e}")

                        response = self._merge_reasoning_branches(response, alternatives, factors_result)

                        overall_score = confidence
                        overall_data = factors_result.get('overall', {})
                        if overall_data and isinstance(overall_data, dict):
                            overall_score = overall_data.get('score', confidence)
                        step = ReasoningStep(
                            phase=ReasoningPhase.FINAL_SYNTHESIS.value,
                            thought=f"Ветвление рассуждений: использовано {len(alternatives)} альтернатив",
                            confidence=overall_score
                        )
                        result.steps.append(step)

                        for alt in alternatives:
                            alt_detail = f"Альтернатива [{alt['factor_name']}]: {alt.get('condition', '')}"
                            if alt.get('generated_response'):
                                alt_detail += f" → {alt['generated_response'][:100]}..."

                            step = ReasoningStep(
                                phase="alternative_branch",
                                thought=alt_detail,
                                confidence=alt.get('priority', 0.5)
                            )
                            result.steps.append(step)
            except Exception as e:
                logger.debug(f"Ошибка анализа факторов: {e}")

            current_threshold = get_adaptive_threshold(iteration)
            if should_terminate(confidence, self.confidence_threshold, iteration):
                logger.info(f"Достаточная уверенность {confidence:.2f} >= {current_threshold:.2f} (итерация {iteration}). Завершаем.")
                result.final_response = response
                break

            if iteration < self.max_iterations:
                questions = self._generate_clarification(analysis, current_query)
                result.clarification_questions = questions

                logger.info(f"Уверенность {confidence:.2f} < {current_threshold:.2f}. Вопросы: {questions}")

                if questions:
                    result.final_response = f"Уточните, пожалуйста: {questions[0]}"
                    break

        if iteration >= self.max_iterations and not result.final_response:
            result.final_response = "Извините, мне нужно больше информации для полного ответа."

        result.processing_time = time.time() - start_time

        logger.info(f"Рассуждение завершено: {iteration} итераций, уверенность {result.confidence:.2f}")

        self._store_reasoning_chain(result)

        return {
            "response": result.final_response,
            "text": result.final_response,
            "status": "ok",
            "confidence": result.confidence,
            "reasoning": result.to_dict(),
            "source": "self_reasoning_engine",
            "processing_time": result.processing_time
        }

    def _analyze_logical_factors(self, query: str, response: str, context: Dict = None) -> Dict[str, Any]:
        factors_result = {}

        factors_result['ethics'] = self._evaluate_ethics_factor(query, response)
        factors_result['knowledge'] = self._evaluate_knowledge_factor(query, response)
        factors_result['contradiction'] = self._evaluate_contradiction_factor(response)
        factors_result['context'] = self._evaluate_context_factor(query, response, context)
        factors_result['logic'] = self._evaluate_logic_factor(query, response)

        total_score = 0
        for factor_name, factor_data in factors_result.items():
            weight = self.LOGICAL_FACTORS.get(factor_name, {}).get('weight', 0.1)
            total_score += factor_data['score'] * weight

        details_copy = {k: v for k, v in factors_result.items()}
        factors_result['overall'] = {
            'score': total_score,
            'details': details_copy
        }

        return factors_result

    def _evaluate_ethics_factor(self, query: str, response: str) -> Dict[str, Any]:
        return {
            'score': 1.0,
            'warnings': [],
            'factor': 'ethics'
        }

    def _evaluate_knowledge_factor(self, query: str, response: str) -> Dict[str, Any]:
        score = 0.8
        sources = []

        if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
            try:
                kg = self.brain.knowledge_graph
                search_method = getattr(kg, 'search_nodes', getattr(kg, 'search', None))
                if search_method:
                    words = query.split()[:5]
                    found = False
                    for word in words:
                        if len(word) > 4:
                            results = search_method(word, limit=1)
                            if results:
                                found = True
                                sources.append(word)
                                break
                    if found:
                        score = 0.9
            except Exception as e:
                logger.debug(f"Ошибка оценки фактора знаний: {e}")

        return {
            'score': score,
            'sources': sources,
            'factor': 'knowledge'
        }

    def _evaluate_contradiction_factor(self, response: str) -> Dict[str, Any]:
        score = 1.0
        contradictions = []

        if self.brain and hasattr(self.brain, 'contradiction_manager'):
            try:
                result = self.brain.contradiction_manager.detect_contradictions(text=response)
                if result and result.get('contradictions'):
                    contradictions = result['contradictions'][:3]
                    score = 1.0 - (len(contradictions) * 0.2)
            except Exception as e:
                logger.debug(f"Ошибка оценки фактора противоречий: {e}")

        return {
            'score': max(0, min(1, score)),
            'contradictions': contradictions,
            'factor': 'contradiction'
        }

    def _evaluate_context_factor(self, query: str, response: str, context: Dict = None) -> Dict[str, Any]:
        score = 0.7

        query_lower = query.lower()
        response_lower = response.lower()

        query_words = [w for w in query_lower.split() if len(w) > 3]
        matched_words = [w for w in query_words if w in response_lower]

        if query_words:
            match_ratio = len(matched_words) / len(query_words)
            score = 0.5 + (match_ratio * 0.5)

        if context and context.get('conversation_history'):
            score = min(1.0, score + 0.1)

        return {
            'score': score,
            'matched_words': matched_words,
            'factor': 'context'
        }

    def _evaluate_logic_factor(self, query: str, response: str) -> Dict[str, Any]:
        score = 1.0
        issues = []

        resp_lower = response.lower()
        query_lower = query.lower()
        resp_sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]

        contradiction_pairs = [
            ('да', 'нет'), ('верно', 'неверно'), ('правильно', 'неправильно'),
            ('возможно', 'невозможно'), ('всегда', 'никогда'), ('все', 'никто'),
            ('можно', 'нельзя'), ('истина', 'ложь'), ('согласен', 'не согласен'),
            ('увеличивается', 'уменьшается'), ('растёт', 'падает'), ('лучше', 'хуже'),
            ('положительно', 'отрицательно'), ('выше', 'ниже'), ('больше', 'меньше'),
        ]

        detected_contradictions = []
        for pos, neg in contradiction_pairs:
            if pos in resp_lower and neg in resp_lower:
                pos_sentences = [s for s in resp_sentences if pos in s.lower()]
                neg_sentences = [s for s in resp_sentences if neg in s.lower()]
                if pos_sentences and neg_sentences:
                    detected_contradictions.append(f'"{pos}" vs "{neg}"')

        if detected_contradictions:
            penalty = min(0.3, len(detected_contradictions) * 0.1)
            score -= penalty
            issues.append(f'Обнаружены противоречия: {", ".join(detected_contradictions)}')

        if '?' in query:
            answer_indicators = ['да,', 'нет,', 'ответ:', 'результат:', 'значит', 'составляет', 'равен']
            has_direct_answer = any(ind in resp_lower for ind in answer_indicators)

            if '?' in query and '?' not in resp_lower and not has_direct_answer:
                if len(response) < 150:
                    score -= 0.15
                    issues.append('Вопрос требует развёрнутого ответа, но ответ неполный')

            query_type = None
            question_words = {
                'почему': 'причинный', 'зачем': 'причинный', 'отчего': 'причинный',
                'как': 'процессуальный', 'каким образом': 'процессуальный',
                'когда': 'временной', 'где': 'пространственный',
                'сколько': 'количественный', 'какой': 'описательный',
                'кто': 'идентификационный', 'что': 'идентификационный',
                'сравни': 'сравнительный', 'чем отличается': 'сравнительный',
            }
            for word, qtype in question_words.items():
                if word in query_lower:
                    query_type = qtype
                    break

            if query_type == 'причинный':
                because_words = ['потому что', 'так как', 'из-за', 'вследствие', 'причина', 'по причине']
                if not any(w in resp_lower for w in because_words):
                    score -= 0.1
                    issues.append('Вопрос о причине, но ответ не содержит причинно-следственной связи')

            elif query_type == 'сравнительный':
                comparison_words = ['отличие', 'разница', 'в отличие', 'сравнение', 'различие', 'по сравнению']
                if not any(w in resp_lower for w in comparison_words):
                    score -= 0.1
                    issues.append('Вопрос требует сравнения, но ответ не содержит сравнительного анализа')

        logical_markers = ['однако', 'но', 'поэтому', 'следовательно', 'значит', 'таким образом']
        for marker in logical_markers:
            if marker in query_lower and marker not in resp_lower:
                if marker in ['поэтому', 'следовательно', 'значит', 'таким образом']:
                    issues.append(f'Отсутствует логическая связка: {marker}')
                    score -= 0.05

        if len(response) < len(query) * 0.5:
            issues.append('Ответ слишком короткий относительно запроса')
            score -= 0.1

        if '?' in query and '?' not in response and len(response) < 100:
            issues.append('Вопрос требовал ответа, но ответ слишком короткий')
            score -= 0.1

        negations = ['не', 'нет', 'нельзя', 'невозможно', 'отсутствует', 'нету']
        double_negatives = 0
        for sent in resp_lower.split('.'):
            neg_count = sum(1 for n in negations if n in sent.split())
            if neg_count >= 2:
                double_negatives += 1
        if double_negatives > 0:
            score -= 0.1
            issues.append(f'Обнаружена двойная отрицательная конструкция ({double_negatives})')

        premise_conclusion_markers = ['если', 'то', 'когда', 'тогда', 'при условии', 'в случае если']
        has_premise = any(m in resp_lower for m in ['если', 'при условии', 'когда'])
        has_conclusion = any(m in resp_lower for m in ['то', 'тогда', 'следует'])
        if has_premise and not has_conclusion:
            score -= 0.1
            issues.append('Ответ содержит предпосылку без вывода')

        return {
            'score': max(0, min(1, score)),
            'issues': issues,
            'factor': 'logic'
        }

    def analyze_response(self, query: str, response: str) -> Dict[str, Any]:
        return self._analyze_logical_factors(query, response)

    def _find_alternative_reasoning_branches(self, query: str, current_response: str, factors_result: Dict) -> List[Dict]:
        alternatives = []

        weak_factors = []
        overall_data = factors_result.get('overall', {})
        details = {}
        if overall_data and isinstance(overall_data, dict):
            details = overall_data.get('details', {})

        if details and isinstance(details, dict):
            for factor_name, factor_data in details.items():
                if not isinstance(factor_data, dict):
                    continue
                if factor_data.get('score', 1.0) < 0.7:
                    weak_factors.append({
                        'factor': factor_name,
                        'score': factor_data.get('score', 0),
                        'issue': factor_data.get('warnings', factor_data.get('contradictions', factor_data.get('issues', [])))
                    })

        for weak in weak_factors:
            factor_name = weak['factor']
            factor_info = self.LOGICAL_FACTORS.get(factor_name, {})

            alt = {
                'factor': factor_name,
                'factor_name': factor_info.get('name', factor_name),
                'current_score': weak['score'],
                'condition': f"ЕСЛИ оценка {factor_name} < 0.7",
                'alternative': self._generate_alternative_for_factor(factor_name, query, current_response),
                'priority': 1.0 - weak['score']
            }
            alternatives.append(alt)

        alternatives.sort(key=lambda x: x['priority'], reverse=True)

        return alternatives[:3]

    def _generate_alternative_for_factor(self, factor_name: str, query: str, current_response: str) -> str:
        if factor_name == 'ethics':
            return f"""Переформулируй следующий ответ, учитывая этические аспекты:
Ответ: {current_response}
Запрос: {query}

Дай этически корректный вариант ответа."""

        elif factor_name == 'knowledge':
            return f"""Проверь и дополни следующий ответ на основе фактических знаний:
Ответ: {current_response}
Запрос: {query}

Дай более точный ответ с фактической информацией."""

        elif factor_name == 'contradiction':
            return f"""Переформулируй ответ, устранив возможные противоречия:
Ответ: {current_response}

Дай непротиворечивый вариант."""

        elif factor_name == 'context':
            return f"""Переформулируй ответ с учетом контекста запроса:
Запрос: {query}
Ответ: {current_response}

Дай ответ, более точно отвечающий на запрос."""

        elif factor_name == 'logic':
            return f"""Проверь логическую согласованность ответа и исправь ошибки:
Ответ: {current_response}

Дай логически согласованный ответ."""

        return f"Пересмотри ответ: {current_response}"

    def _should_use_alternative_branch(self, factors_result: Dict, threshold: float = 0.7) -> bool:
        overall_data = factors_result.get('overall', {})
        overall = 1.0
        if overall_data and isinstance(overall_data, dict):
            overall = overall_data.get('score', 1.0)

        if overall < threshold + 0.1:
            return True

        details = {}
        if overall_data and isinstance(overall_data, dict):
            details = overall_data.get('details', {})

        if details and isinstance(details, dict):
            for factor_data in details.values():
                if isinstance(factor_data, dict) and factor_data.get('score', 1.0) < threshold:
                    return True

        return False

    def _merge_reasoning_branches(self, primary_response: str, alternatives: List[Dict], factors_result: Dict) -> str:
        if not alternatives:
            return primary_response

        final_parts = [primary_response]

        for alt in alternatives:
            if alt.get('alternative'):
                final_parts.append(f"\n\n[Альтернативный вариант ({alt['factor_name']})]: {alt['alternative']}")

        return "\n".join(final_parts)

    def _analyze_response(self, query: str, response: str) -> AnalysisResult:
        analysis = AnalysisResult()

        try:
            if hasattr(self.brain, 'ethics_framework'):
                ethics = self.brain.ethics_framework
                if hasattr(ethics, 'analyze_response'):
                    analysis.ethics_result = ethics.analyze_response(query, response)
        except Exception as e:
            logger.warning(f"Ethics check failed: {e}")

        try:
            if hasattr(self.brain, 'contradiction_manager'):
                contr = self.brain.contradiction_manager
                if hasattr(contr, 'detect_contradictions'):
                    analysis.contradiction_result = contr.detect_contradictions(text=response)
        except Exception as e:
            logger.warning(f"Contradiction check failed: {e}")

        try:
            if hasattr(self.brain, 'knowledge_graph'):
                kg = self.brain.knowledge_graph
                if hasattr(kg, 'search'):
                    analysis.knowledge_result = {"coverage": {"score": 0.5}}
        except Exception as e:
            logger.warning(f"Knowledge check failed: {e}")

        return analysis

    def _generate_clarification(self, analysis: AnalysisResult, query: str) -> List[str]:
        try:
            analysis_dict = analysis.to_dict()
            questions = self.clarification_gen.generate_clarification(analysis_dict, query)
            return questions if questions else self.clarification_gen.generate_simple_clarification(query)
        except Exception as e:
            logger.warning(f"Clarification generation failed: {e}")
            return ["Можете уточнить ваш запрос?"]

    def _store_reasoning_chain(self, result: ReasoningResult) -> None:
        if not self.fractal_storage:
            self.fractal_storage = getattr(self.brain, 'fractal_storage', None)

        if not self.fractal_storage:
            return

        try:
            for i, step in enumerate(result.steps):
                self.fractal_storage.add_reasoning_step(
                    query=result.query,
                    step_content=step.thought,
                    confidence=step.confidence,
                    iteration=i + 1
                )

            logger.debug(f"Сохранено {len(result.steps)} шагов рассуждения в FractalStorage")
        except Exception as e:
            logger.warning(f"Не удалось сохранить цепочку рассуждений: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_queries': self.total_queries,
            'total_iterations': self.total_iterations,
            'confidence_threshold': self.confidence_threshold,
            'max_iterations': self.max_iterations
        }

    # Methods imported from sre_context, sre_quality, sre_recursive:
    _build_contextual_query = _build_contextual_query
    _determine_query_type = _determine_query_type
    _check_response_quality = check_quality
    _init_retriever = _init_retriever
    _get_generation_params = _get_generation_params
    _is_complex_query = _is_complex_query
    _check_relevance = _check_relevance
    _generate_with_qwen = _generate_with_qwen

    def health_check(self) -> Dict[str, Any]:
        checks = {
            "engine_initialized": True,
            "fractal_storage_ok": self.fractal_storage is not None,
            "fractal_embedder_ok": self.fractal_embedder is not None,
            "fractal_retriever_ok": self.fractal_retriever is not None,
            "qwen_cached": self._qwen_cached is not None
        }

        healthy = all(v for v in checks.values())

        return {
            "healthy": healthy,
            "checks": checks,
            "stats": self.get_stats()
        }


def create_reasoning_engine(brain, config: Optional[Dict] = None) -> SelfReasoningEngine:
    """Фабричная функция для создания движка"""
    return SelfReasoningEngine(brain=brain, config=config)

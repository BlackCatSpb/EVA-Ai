"""
Self-Reasoning Engine - главный движок рассуждения ЕВА
Цикл: Generate → Analyze → Clarify → Repeat until confidence >= 0.75
"""

import time
import logging
from typing import Dict, Any, Optional, List

from .reasoning_types import (
    ReasoningStep, 
    ReasoningResult, 
    AnalysisResult,
    ReasoningPhase
)
from .confidence_scorer import calculate_overall_confidence, should_terminate, CONFIDENCE_THRESHOLD
from .clarification_generator import ClarificationGenerator

logger = logging.getLogger(__name__)


# Параметры из DESIGN.md
MAX_ITERATIONS = 5
DEFAULT_MAX_NEW_TOKENS = 2048
MAX_RECURSION_DEPTH = 3


class SelfReasoningEngine:
    """
    Движок самостоятельного рассуждения с поддержкой рекурсии
    Использует ЕДИНСТВЕННЫЙ экземпляр Qwen (singleton) для генерации
    """
    
    def __init__(self, brain, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация движка рассуждения
        
        Args:
            brain: CoreBrain instance с qwen_model_manager (singleton)
            config: Конфигурация {max_iterations, confidence_threshold}
        """
        self.brain = brain
        self.config = config or {}
        
        # Параметры
        self.max_iterations = self.config.get('max_iterations', MAX_ITERATIONS)
        self.confidence_threshold = self.config.get('confidence_threshold', CONFIDENCE_THRESHOLD)
        self.max_new_tokens = self.config.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
        self.max_recursion_depth = self.config.get('max_recursion_depth', MAX_RECURSION_DEPTH)
        
        # Компоненты
        self.clarification_gen = ClarificationGenerator()
        
        # Fractal Storage для хранения цепочек рассуждений
        self.fractal_storage = getattr(brain, 'fractal_storage', None) if brain else None
        
        # Fallback: создаём FractalStorage если его нет в brain
        if self.fractal_storage is None:
            try:
                from eva.reasoning.fractal_ml import FractalStorage
                storage_path = './cache/fractal_reasoning'
                if brain and hasattr(brain, 'cache_dir'):
                    import os
                    storage_path = os.path.join(brain.cache_dir, 'fractal_reasoning')
                self.fractal_storage = FractalStorage(storage_dir=storage_path)
                logger.info(f"FractalStorage создан: {storage_path}")
            except Exception as e:
                logger.warning(f"Не удалось создать FractalStorage: {e}")
                self.fractal_storage = None
        
        # Fractal компоненты для рекурсивного reasoning
        self.fractal_embedder = None
        self.fractal_retriever = None
        self._init_fractal_components()
        
        # Кэш для Qwen - избегаем повторной инициализации
        self._qwen_cached = None
        
        # Статистика
        self.total_queries = 0
        self.total_iterations = 0
        self.recursive_calls = 0
        
        logger.info(f"SelfReasoningEngine инициализирован: max_iterations={self.max_iterations}, recursion_depth={self.max_recursion_depth}")
    
    def _init_fractal_components(self):
        """Инициализация FractalRetriever и FractalEmbedder"""
        try:
            from eva.reasoning.fractal_ml.fractal_embedder import FractalEmbedder
            from eva.reasoning.fractal_ml.fractal_retriever import FractalRetriever
            
            self.fractal_embedder = FractalEmbedder(use_sentence_transformers=False)
            self.fractal_retriever = None
            logger.info("FractalEmbedder инициализирован")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать fractal компоненты: {e}")
    
    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Обработка запроса через цикл самостоятельного рассуждения
        
        Args:
            query: Запрос пользователя
            user_context: Дополнительный контекст (может содержать conversation_history)
            
        Returns:
            Dict с полями: response, reasoning (для GUI), confidence, iterations
        """
        start_time = time.time()
        self.total_queries += 1
        
        # Извлекаем историю диалогов из контекста
        conversation_history = []
        if user_context and 'conversation_history' in user_context:
            conversation_history = user_context['conversation_history']
            logger.info(f"Получена история диалогов: {len(conversation_history)} сообщений")
        
        # Формируем расширенный промпт с историей
        enhanced_query = self._build_contextual_query(query, conversation_history)
        
        # Быстрая проверка - сразу используем fallback если модели недоступны
        try:
            qwen = getattr(self.brain, 'qwen_model_manager', None)
            if qwen is None or not getattr(qwen, 'initialized', False):
                # Qwen недоступен - используем простой ответ напрямую
                simple_response = self._generate_simple_response(enhanced_query)
                return {
                    "response": simple_response,
                    "text": simple_response,
                    "status": "ok",
                    "confidence": 0.5,
                    "reasoning": {"source": "simple_fallback"},
                    "source": "self_reasoning_engine",
                    "processing_time": time.time() - start_time,
                    "conversation_history_used": len(conversation_history) > 0
                }
        except Exception as e:
            logger.warning(f"Error in process_query fallback: {e}")
        
        logger.info(f"Начинаем рассуждение для запроса: {query[:50]}...")
        
        # Проверяем сложность запроса - используем рекурсию для сложных
        self._init_retriever()
        
        if self.fractal_retriever and self._is_complex_query(query):
            logger.info("Сложный запрос - используем рекурсивный reasoning")
            result = self._recursive_process_query(query, user_context, depth=0)
            result["processing_time"] = time.time() - start_time
            return result
        
        # Инициализация результата
        result = ReasoningResult(
            final_response="",
            confidence=0.0,
            iterations=0,
            query=query
        )
        
        current_query = query
        iteration = 0
        
        # === ОСНОВНОЙ ЦИКЛ РАССУЖДЕНИЯ ===
        while iteration < self.max_iterations:
            iteration += 1
            result.iterations = iteration
            self.total_iterations += 1
            
            logger.info(f"Итерация {iteration}/{self.max_iterations}")
            
            # Шаг 1: Генерация ответа через Qwen (singleton!)
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
            
            # Шаг 2: Анализ ответа
            analysis = self._analyze_response(current_query, response)
            
            # Шаг 3: Расчёт уверенности
            confidence = calculate_overall_confidence(
                ethics_result=analysis.ethics_result,
                contradiction_result=analysis.contradiction_result,
                knowledge_result=analysis.knowledge_result,
                query=current_query
            )
            
            step = ReasoningStep(
                phase=ReasoningPhase.FINAL_SYNTHESIS.value,
                thought=f"Анализ завершён. Уверенность: {confidence:.2f}",
                confidence=confidence
            )
            result.steps.append(step)
            
            result.confidence = confidence
            
            # === ВЕТВЛЕНИЕ РАССУЖДЕНИЙ ===
            # Анализируем логические факторы
            try:
                factors_result = self._analyze_logical_factors(current_query, response, user_context)
                
                # Добавляем шаг анализа факторов
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
                
                # Добавляем шаг с анализом факторов в цепочку
                step = ReasoningStep(
                    phase="logical_analysis",
                    thought=f"Анализ факторов: {', '.join(factor_summary)}",
                    confidence=factors_result.get('overall', {}).get('score', confidence)
                )
                result.steps.append(step)
                
                # Проверяем условие для альтернативной ветви
                if self._should_use_alternative_branch(factors_result):
                    logger.info("Условие выполнено: используем альтернативную ветвь рассуждения")
                    
                    # Находим альтернативы
                    alternatives = self._find_alternative_reasoning_branches(current_query, response, factors_result)
                    
                    if alternatives:
                        # Логируем найденные альтернативы
                        for alt in alternatives:
                            logger.info(f"Альтернатива: {alt['factor_name']} (приоритет: {alt['priority']:.2f})")
                        
                        # Генерируем альтернативные ответы
                        for alt in alternatives:
                            try:
                                alt_response = self._generate_with_qwen(alt['alternative'])
                                alt['generated_response'] = alt_response
                            except Exception as e:
                                logger.debug(f"Ошибка генерации альтернативы: {e}")
                        
                        # Объединяем результаты
                        response = self._merge_reasoning_branches(response, alternatives, factors_result)
                        
                        # Добавляем информацию о ветвлении в шаг
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
                        
                        # Добавляем детали каждой альтернативы как отдельные шаги
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
            
            # Шаг 4: Проверка на завершение
            if should_terminate(confidence, self.confidence_threshold):
                logger.info(f"Достаточная уверенность {confidence:.2f} >= {self.confidence_threshold}. Завершаем.")
                result.final_response = response
                break
            
            # Шаг 5: Генерация уточняющих вопросов
            if iteration < self.max_iterations:
                questions = self._generate_clarification(analysis, current_query)
                result.clarification_questions = questions
                
                logger.info(f"Уверенность {confidence:.2f} < {self.confidence_threshold}. Вопросы: {questions}")
                
                # Задаём первый вопрос пользователю (в качестве ответа)
                if questions:
                    result.final_response = f"Уточните, пожалуйста: {questions[0]}"
                    break
        
        # Если достигли максимума итераций
        if iteration >= self.max_iterations and not result.final_response:
            result.final_response = "Извините, мне нужно больше информации для полного ответа."
        
        result.processing_time = time.time() - start_time
        
        logger.info(f"Рассуждение завершено: {iteration} итераций, уверенность {result.confidence:.2f}")
        
        # Сохраняем цепочку рассуждений в FractalStorage если доступен
        self._store_reasoning_chain(result)
        
        # Возвращаем результат в формате для CoreBrain
        return {
            "response": result.final_response,
            "text": result.final_response,
            "status": "ok",
            "confidence": result.confidence,
            "reasoning": result.to_dict(),  # Для GUI панели
            "source": "self_reasoning_engine",
            "processing_time": result.processing_time
        }
    
    def _build_contextual_query(self, query: str, conversation_history: List[Dict]) -> str:
        """
        Формирует расширенный промпт с историей диалогов.
        
        Args:
            query: Текущий запрос пользователя
            conversation_history: История предыдущих сообщений
            
        Returns:
            Расширенный промпт с контекстом
        """
        if not conversation_history:
            return query
        
        # Берем последние 5 сообщений для контекста
        recent_history = conversation_history[-10:]
        
        context_parts = []
        for msg in recent_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if content:
                role_label = 'Пользователь' if role == 'user' else 'Ассистент'
                context_parts.append(f"{role_label}: {content[:200]}")
        
        if not context_parts:
            return query
        
        # Формируем расширенный промпт
        context_str = "\n".join(context_parts)
        enhanced = f"""Предыдущий контекст разговора:
{context_str}

Текущий вопрос: {query}

Дай ответ с учетом контекста предыдущего разговора."""
        
        logger.info(f"Сформирован расширенный промпт с {len(recent_history)} сообщениями истории")
        return enhanced
    
    # === Логическое рассуждение с факторами ===
    
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
    
    def _analyze_logical_factors(self, query: str, response: str, context: Dict = None) -> Dict[str, Any]:
        """
        Анализирует ответ по всем логическим факторам системы.
        
        Returns:
            Dict с оценками каждого фактора и рекомендациями
        """
        factors_result = {}
        
        # Фактор 1: Этика
        factors_result['ethics'] = self._evaluate_ethics_factor(query, response)
        
        # Фактор 2: Знания
        factors_result['knowledge'] = self._evaluate_knowledge_factor(query, response)
        
        # Фактор 3: Противоречия
        factors_result['contradiction'] = self._evaluate_contradiction_factor(response)
        
        # Фактор 4: Контекст
        factors_result['context'] = self._evaluate_context_factor(query, response, context)
        
        # Фактор 5: Логика
        factors_result['logic'] = self._evaluate_logic_factor(query, response)
        
        # Рассчитываем общую оценку
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
        """Оценка этического фактора"""
        score = 1.0
        warnings = []
        
        # Проверяем через ethics_checker если доступен
        if self.brain and hasattr(self.brain, 'ethics_checker'):
            try:
                result = self.brain.ethics_checker.check(response)
                if result and result.get('warnings'):
                    warnings = result['warnings']
                    score = 1.0 - (len(warnings) * 0.1)
            except Exception as e:
                logger.debug(f"Ошибка оценки этического фактора: {e}")
        
        return {
            'score': max(0, min(1, score)),
            'warnings': warnings,
            'factor': 'ethics'
        }
    
    def _evaluate_knowledge_factor(self, query: str, response: str) -> Dict[str, Any]:
        """Оценка фактора знаний"""
        score = 0.8  # Базовый балл
        sources = []
        
        # Проверяем через knowledge_graph
        if self.brain and hasattr(self.brain, 'knowledge_graph') and self.brain.knowledge_graph:
            try:
                kg = self.brain.knowledge_graph
                # Ищем релевантную информацию
                search_method = getattr(kg, 'search_nodes', getattr(kg, 'search', None))
                if search_method:
                    # Простой поиск по ключевым словам из ответа
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
        """Оценка фактора противоречий"""
        score = 1.0
        contradictions = []
        
        # Проверяем через contradiction_manager
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
        """Оценка фактора контекста"""
        score = 0.7
        
        # Проверяем, отвечает ли ответ на запрос
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Простая проверка - есть ли ключевые слова запроса в ответе
        query_words = [w for w in query_lower.split() if len(w) > 3]
        matched_words = [w for w in query_words if w in response_lower]
        
        if query_words:
            match_ratio = len(matched_words) / len(query_words)
            score = 0.5 + (match_ratio * 0.5)
        
        # Учитываем историю диалога
        if context and context.get('conversation_history'):
            score = min(1.0, score + 0.1)
        
        return {
            'score': score,
            'matched_words': matched_words,
            'factor': 'context'
        }
    
    def _evaluate_logic_factor(self, query: str, response: str) -> Dict[str, Any]:
        """Оценка логического фактора"""
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
        """Анализ ответа по логическим факторам"""
        return self._analyze_logical_factors(query, response)
    
    def _find_alternative_reasoning_branches(self, query: str, current_response: str, factors_result: Dict) -> List[Dict]:
        """
        Находит альтернативные ветви рассуждения на основе слабых факторов.
        
        Returns:
            Список альтернативных ветвей с условиями
        """
        alternatives = []
        
        # Анализируем факторы с низкими оценками
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
        
        # Генерируем альтернативы для слабых факторов
        for weak in weak_factors:
            factor_name = weak['factor']
            factor_info = self.LOGICAL_FACTORS.get(factor_name, {})
            
            alt = {
                'factor': factor_name,
                'factor_name': factor_info.get('name', factor_name),
                'current_score': weak['score'],
                'condition': f"ЕСЛИ оценка {factor_name} < 0.7",
                'alternative': self._generate_alternative_for_factor(factor_name, query, current_response),
                'priority': 1.0 - weak['score']  # Приоритет = чем ниже оценка, тем выше
            }
            alternatives.append(alt)
        
        # Сортируем по приоритету
        alternatives.sort(key=lambda x: x['priority'], reverse=True)
        
        return alternatives[:3]  # Максимум 3 альтернативы
    
    def _generate_alternative_for_factor(self, factor_name: str, query: str, current_response: str) -> str:
        """Генерирует промпт для альтернативной ветви рассуждения"""
        
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
        """
        Определяет, нужно ли использовать альтернативную ветвь рассуждения.
        
        Conditions:
        - ЕСЛИ хотя бы один фактор < threshold
        - ИЛИ ЕСЛИ общая оценка < threshold + 0.1
        """
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
        """
        Объединяет результаты основной и альтернативных ветвей рассуждения.
        """
        if not alternatives:
            return primary_response
        
        # Формируем финальный ответ с учётом альтернатив
        final_parts = [primary_response]
        
        for alt in alternatives:
            if alt.get('alternative'):
                final_parts.append(f"\n\n[Альтернативный вариант ({alt['factor_name']})]: {alt['alternative']}")
        
        return "\n".join(final_parts)
    
    def _generate_with_qwen(self, prompt: str) -> str:
        """
        Генерация ответа с fallback на разные модели
        Приоритет: Qwen → FractalModelManager → ResponseGenerator
        """
        # Попытка 1: Qwen singleton (используем кэш)
        try:
            if self._qwen_cached is None:
                if self.brain is not None:
                    qwen = getattr(self.brain, 'qwen_model_manager', None)
                else:
                    qwen = None
                
                if qwen is None or not getattr(qwen, 'initialized', False):
                    try:
                        from eva.mlearning.qwen_model_manager import get_qwen_model_manager
                        qwen = get_qwen_model_manager(
                            model_size='qwen3.5-0.8b',
                            device='auto',
                            load_in_8bit=True
                        )
                        # Кэшируем только успешно инициализированный объект
                        if qwen is not None:
                            self._qwen_cached = qwen
                    except Exception as e:
                        logger.warning(f"Failed to initialize Qwen: {e}")
                        # Не кэшируем неудачную попытку - оставляем None для повтора
            
            if self._qwen_cached is not None and getattr(self._qwen_cached, 'initialized', False):
                messages = [{"role": "user", "content": prompt}]
                response = self._qwen_cached.generate(
                    messages,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1
                )
                if response:
                    return response
        except Exception as e:
            logger.debug(f"Qwen generation failed: {e}")
        
        # Попытка 2: FractalModelManager
        try:
            if self.brain is not None:
                fractal_mm = getattr(self.brain, 'fractal_model_manager', None)
                if fractal_mm and hasattr(fractal_mm, 'generate_response'):
                    response = fractal_mm.generate_response(prompt)
                    if response:
                        return response
        except Exception as e:
            logger.debug(f"FractalModelManager generation failed: {e}")
        
        # Попытка 3: ResponseGenerator
        try:
            if self.brain is not None:
                resp_gen = getattr(self.brain, 'response_generator', None)
                if resp_gen and hasattr(resp_gen, 'generate'):
                    result = resp_gen.generate(prompt)
                    if result and isinstance(result, dict):
                        return result.get('text', result.get('response', ''))
                    elif result:
                        return str(result)
        except Exception as e:
            logger.debug(f"ResponseGenerator generation failed: {e}")
        
        # Попытка 4: GenerationCoordinator
        try:
            if self.brain is not None:
                gen_coord = getattr(self.brain, 'generation_coordinator', None)
                if gen_coord and hasattr(gen_coord, 'generate'):
                    result = gen_coord.generate(text=prompt, source="reasoning_engine")
                    if result:
                        return result.text if hasattr(result, 'text') else str(result)
        except Exception as e:
            logger.debug(f"GenerationCoordinator failed: {e}")
        
        # Fallback: простой ответ без модели
        logger.warning("Все генераторы недоступны, используем простой ответ")
        return self._generate_simple_response(prompt)
    
    def _generate_simple_response(self, prompt: str) -> str:
        """Простой fallback ответ без модели."""
        prompt_lower = prompt.lower().strip()
        
        # Простые шаблоны для приветствий
        greetings = ['привет', 'здравствуй', 'hello', 'hi', 'хай', 'прив', 'здорово']
        if any(g in prompt_lower for g in greetings):
            return "Здравствуйте! Я ЕВА. Чем могу помочь?"
        
        # Попробуем использовать knowledge graph если доступен
        kg_response = self._get_knowledge_response(prompt_lower)
        if kg_response:
            return kg_response
        
        # Ключевые слова для тематических ответов
        keyword_responses = {
            'погода': 'Для информации о погоде я могу выполнить поиск в интернете.',
            'новост': 'Могу найти последние новости по вашему запросу.',
            'помощ': 'Я могу помочь с ответами на вопросы, анализом информации и поиском данных.',
            'что такое': 'Для объяснения понятий мне нужно больше контекста.',
            'как работает': 'Могу объяснить принципы работы, но для точного ответа уточните область.',
            'кто такой': 'Для идентификации личности нужны дополнительные детали.',
            'спасиб': 'Пожалуйста! Рад был помочь.',
            'благодар': 'Спасибо! Обращайтесь ещё.',
            'пока': 'До свидания! Возвращайтесь с новыми вопросами.',
            'помоги': 'Опишите подробнее, что именно вам нужно, и я постараюсь помочь.',
        }
        
        for keyword, response in keyword_responses.items():
            if keyword in prompt_lower:
                return response
        
        # Простые ответы на общие вопросы
        if '?' in prompt:
            return f"Интересный вопрос: '{prompt}'. Для полного ответа требуется больше контекста."
        
        return f"Получил ваш запрос: '{prompt}'. Чтобы дать точный ответ, уточните детали."
    
    def _get_knowledge_response(self, prompt: str) -> Optional[str]:
        """Попытка получить ответ из knowledge graph"""
        try:
            if self.brain is not None and hasattr(self.brain, 'knowledge_graph'):
                kg = self.brain.knowledge_graph
                if kg is None:
                    return None
                # Try search_nodes first (proper method)
                search_method = getattr(kg, 'search_nodes', getattr(kg, 'search', getattr(kg, 'search_by_concept', None)))
                if search_method:
                    results = search_method(prompt, limit=3)
                    if results and isinstance(results, list) and len(results) > 0:
                        best = results[0]
                        if isinstance(best, dict):
                            content = best.get('content', best.get('text', ''))
                            if content:
                                return f"Известно: {content[:200]}..."
                        else:
                            content = getattr(best, 'content', getattr(best, 'description', ''))
                            if content:
                                return f"Известно: {content[:200]}..."
        except Exception as e:
            logger.debug(f"Knowledge search error: {e}")
        return None
    
    def _analyze_response(self, query: str, response: str) -> AnalysisResult:
        """
        Анализ ответа через системные модули
        """
        analysis = AnalysisResult()
        
        # Ethics check
        try:
            if hasattr(self.brain, 'ethics_framework'):
                ethics = self.brain.ethics_framework
                if hasattr(ethics, 'analyze_response'):
                    analysis.ethics_result = ethics.analyze_response(query, response)
        except Exception as e:
            logger.warning(f"Ethics check failed: {e}")
        
        # Contradiction check
        try:
            if hasattr(self.brain, 'contradiction_manager'):
                contr = self.brain.contradiction_manager
                if hasattr(contr, 'detect_contradictions'):
                    analysis.contradiction_result = contr.detect_contradictions(text=response)
        except Exception as e:
            logger.warning(f"Contradiction check failed: {e}")
        
        # Knowledge check
        try:
            if hasattr(self.brain, 'knowledge_graph'):
                kg = self.brain.knowledge_graph
                if hasattr(kg, 'search'):
                    # Простой поиск по графу знаний
                    analysis.knowledge_result = {"coverage": {"score": 0.5}}
        except Exception as e:
            logger.warning(f"Knowledge check failed: {e}")
        
        return analysis
    
    def _generate_clarification(self, analysis: AnalysisResult, query: str) -> List[str]:
        """
        Генерация уточняющих вопросов
        """
        try:
            # Используем анализ + исходный запрос
            analysis_dict = analysis.to_dict()
            questions = self.clarification_gen.generate_clarification(analysis_dict, query)
            return questions if questions else self.clarification_gen.generate_simple_clarification(query)
        except Exception as e:
            logger.warning(f"Clarification generation failed: {e}")
            return ["Можете уточнить ваш запрос?"]
    
    def _store_reasoning_chain(self, result: ReasoningResult) -> None:
        """
        Сохранение цепочки рассуждений в FractalStorage
        """
        if not self.fractal_storage:
            # Пробуем получить из brain
            self.fractal_storage = getattr(self.brain, 'fractal_storage', None)
        
        if not self.fractal_storage:
            return
        
        try:
            # Сохраняем каждый шаг рассуждения
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
    
    # =====================================================
    # === РЕКУРСИВНЫЕ МЕТОДЫ ===
    # =====================================================
    
    def _init_retriever(self):
        """Инициализация Retriever после подключения storage"""
        if self.fractal_storage and not self.fractal_retriever and self.fractal_embedder:
            try:
                from eva.reasoning.fractal_ml.fractal_retriever import FractalRetriever
                self.fractal_retriever = FractalRetriever(
                    storage=self.fractal_storage,
                    embedder=self.fractal_embedder
                )
                logger.info("FractalRetriever инициализирован")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать Retriever: {e}")
    
    def _is_complex_query(self, query: str) -> bool:
        """Определить сложность запроса для выбора стратегии"""
        complexity_indicators = [
            " и ", " или ", " но ", " потому что ", " поэтому",
            " если ", " тогда ", " следовательно ", " значит ",
            " как ", "почему", "какие", "сравни", "различия", "отличие"
        ]
        
        query_lower = query.lower()
        complexity_score = sum(1 for ind in complexity_indicators if ind in query_lower)
        
        is_long = len(query.split()) > 15
        
        return complexity_score >= 2 or is_long
    
    def _recursive_process_query(
        self, 
        query: str, 
        user_context: Optional[Dict],
        depth: int
    ) -> Dict[str, Any]:
        """Рекурсивная обработка сложных запросов"""
        self.recursive_calls += 1
        
        logger.info(f"Рекурсивный вызов depth={depth}, query={query[:30]}...")
        
        if depth >= self.max_recursion_depth:
            logger.info(f"Достигнута максимальная глубина {self.max_recursion_depth}")
            return self._linear_process_query(query, user_context)
        
        self._init_retriever()
        
        sub_queries = self.decompose_query(query)
        logger.info(f"Декомпозиция на {len(sub_queries)} подзадач")
        
        # Если декомпозиция не удалась - используем линейную обработку
        if not sub_queries:
            logger.info("Декомпозиция вернула пустой список, используем линейную обработку")
            return self._linear_process_query(query, user_context)
        
        sub_results = []
        for sq in sub_queries:
            if depth + 1 <= self.max_recursion_depth:
                sub_result = self._recursive_process_query(sq, user_context, depth=depth + 1)
            else:
                sub_result = self._linear_process_query(sq, user_context)
            sub_results.append(sub_result)
        
        similar_reasoning = []
        if self.fractal_retriever:
            similar_reasoning = self.retrieve_similar_reasoning(query)
            logger.info(f"Найдено {len(similar_reasoning)} похожих рассуждений")
        
        final_result = self._synthesize_recursive_results(
            query, sub_results, similar_reasoning, depth
        )
        
        return final_result
    
    def decompose_query(self, query: str) -> List[str]:
        """Декомпозиция сложного запроса на подзадачи"""
        prompt = f"""Разбей запрос на 2-4 простых подзапроса.
Запрос: {query}
Верни только список подзапросов, каждый на новой строке:"""
        
        try:
            response = self._generate_with_qwen(prompt)
            if response:
                sub_queries = [
                    line.strip() for line in response.split('\n')
                    if line.strip() and len(line.strip()) > 10
                ]
                if len(sub_queries) >= 2:
                    logger.info(f"Декомпозиция успешна: {len(sub_queries)} подзадач")
                    return sub_queries
                else:
                    logger.info(f"Декомпозиция вернула мало подзадач: {len(sub_queries)}, использую линейный режим")
        except Exception as e:
            logger.warning(f"Декомпозиция не удалась: {e}")
        
        # Возвращаем пустой список - будет использован линейный режим
        logger.info(f"Декомпозиция вернула пустой список для: {query[:30]}...")
        return []
    
    def retrieve_similar_reasoning(self, query: str) -> List[Dict]:
        """Найти похожие рассуждения из прошлого"""
        if not self.fractal_retriever or not self.fractal_storage:
            return []
        
        try:
            results = self.fractal_retriever.retrieve_with_embedding(query=query, top_k=5)
            similar = [r for r in results if r.get('node_type') == 'reasoning_step']
            return similar
        except Exception as e:
            logger.warning(f"Ошибка поиска похожих рассуждений: {e}")
            return []
    
    def build_recursive_context(self, query: str) -> Dict:
        """Построить контекст из разных уровней хранилища"""
        context = {"level_0": [], "level_1": [], "level_2": [], "level_3": [], "similar": []}
        
        if not self.fractal_retriever:
            return context
        
        try:
            cross_level = self.fractal_retriever.retrieve_cross_level(query=query, levels=[0, 1, 2, 3])
            for level, nodes in cross_level.items():
                context[f"level_{level}"] = nodes
            context["similar"] = self.retrieve_similar_reasoning(query)
        except Exception as e:
            logger.warning(f"Ошибка построения контекста: {e}")
        
        return context
    
    def _synthesize_recursive_results(
        self,
        query: str,
        sub_results: List[Dict],
        similar_reasoning: List[Dict],
        depth: int
    ) -> Dict[str, Any]:
        """Синтез результатов рекурсивной обработки с consistent confidence calculation"""
        combined_responses = [r.get("response", "") for r in sub_results]
        combined_confidences = [r.get("confidence", 0.0) for r in sub_results]
        
        # Consistent confidence calculation using weighted average
        if combined_confidences:
            # Weight by number of iterations (more iterations = higher confidence)
            weights = [r.get("iterations", 1) for r in sub_results]
            total_weight = sum(weights)
            avg_confidence = sum(c * w for c, w in zip(combined_confidences, weights)) / total_weight if total_weight > 0 else 0.5
        else:
            avg_confidence = 0.5
        
        # Apply consistent boost from similar reasoning
        if similar_reasoning:
            similar_confidences = [s.get("confidence", 0.5) for s in similar_reasoning]
            if similar_confidences:
                avg_similar = sum(similar_confidences) / len(similar_confidences)
                # Apply capped boost (max 0.1)
                boost = min(0.1, avg_similar * 0.2)
                avg_confidence = min(1.0, avg_confidence + boost)
        
        # Ensure confidence is always within valid range and properly rounded
        avg_confidence = round(max(0.0, min(1.0, avg_confidence)), 3)
        
        prompt = f"""На основе подответов составь единый ответ на вопрос.
        Вопрос: {query}
        Подответы: {' '.join(combined_responses)}
        Дай финальный ответ:"""
        
        final_response = self._generate_with_qwen(prompt)
        
        return {
            "response": final_response or (combined_responses[0] if combined_responses else ""),
            "text": final_response or (combined_responses[0] if combined_responses else ""),
            "status": "ok",
            "confidence": avg_confidence,
            "source": "recursive_reasoning",
            "recursive_depth": depth + 1,
            "sub_queries_processed": len(sub_results),
            "similar_found": len(similar_reasoning),
            "reasoning": {
                "sub_results": sub_results,
                "similar_reasoning": similar_reasoning,
                "depth": depth
            }
        }
    
    def _linear_process_query(
        self, 
        query: str, 
        user_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Стандартная линейная обработка (без рекурсии)"""
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
            
            step = ReasoningStep(
                phase=ReasoningPhase.GENERATION.value,
                thought=f"Генерирую ответ на: {current_query[:30]}...",
                confidence=0.0
            )
            
            response = self._generate_with_qwen(current_query)
            result.steps.append(step)
            
            if not response or response.startswith("Ошибка"):
                result.final_response = response or "Ошибка генерации"
                break
            
            analysis = self._analyze_response(current_query, response)
            
            confidence = calculate_overall_confidence(
                ethics_result=analysis.ethics_result,
                contradiction_result=analysis.contradiction_result,
                knowledge_result=analysis.knowledge_result,
                query=current_query
            )
            
            step = ReasoningStep(
                phase=ReasoningPhase.FINAL_SYNTHESIS.value,
                thought=f"Анализ завершён. Уверенность: {confidence:.2f}",
                confidence=confidence
            )
            result.steps.append(step)
            
            result.confidence = confidence
            
            if should_terminate(confidence, self.confidence_threshold):
                result.final_response = response
                break
            
            if iteration < self.max_iterations:
                questions = self._generate_clarification(analysis, current_query)
                result.clarification_questions = questions
                
                if questions:
                    result.final_response = f"Уточните, пожалуйста: {questions[0]}"
                    break
        
        if iteration >= self.max_iterations and not result.final_response:
            result.final_response = "Извините, мне нужно больше информации."
        
        self._store_reasoning_chain(result)
        
        return {
            "response": result.final_response,
            "text": result.final_response,
            "status": "ok",
            "confidence": result.confidence,
            "reasoning": result.to_dict(),
            "source": "linear_reasoning",
            "processing_time": 0.0
        }
    
    # =====================================================
    # === FEEDBACK LOOP МЕТОДЫ (ДЛЯ САМООБУЧЕНИЯ) ===
    # =====================================================
    
    def process_user_feedback(self, query: str, feedback: str, rating: float) -> Dict[str, Any]:
        """
        Обработать фидбек пользователя для самообучения
        
        Args:
            query: Оригинальный запрос
            feedback: Текст обратной связи
            rating: Оценка от 0.0 до 1.0
            
        Returns:
            Dict с результатами обработки
        """
        try:
            # Находим цепочку рассуждений для запроса
            if self.fractal_storage:
                chain = self.fractal_storage.get_reasoning_chain(query)
                
                # Обновляем уверенность на основе рейтинга
                for node in chain:
                    old_conf = node.context.get("confidence", 0.5) if node.context else 0.5
                    # Корректируем уверенность: учитываем фидбек
                    adjusted_conf = (old_conf + rating) / 2
                    
                    if node.context:
                        node.context["confidence"] = adjusted_conf
                        node.context["last_feedback_rating"] = rating
                        node.context["feedback_time"] = time.time()
                
                self.fractal_storage._save()
                
                # Триггерируем самообучение если рейтинг низкий
                if rating < 0.3:
                    self._trigger_self_learning(query, feedback)
                
                logger.info(f"Обработан фидбек для '{query[:30]}...': rating={rating:.2f}")
                
                return {
                    "status": "processed",
                    "chain_updated": len(chain),
                    "triggered_learning": rating < 0.3
                }
            
            return {"status": "no_storage", "message": "FractalStorage не доступен"}
            
        except Exception as e:
            logger.error(f"Ошибка обработки фидбека: {e}")
            return {"status": "error", "message": str(e)}
    
    def refine_reasoning_chain(self, node_id: str, correction: str) -> Dict[str, Any]:
        """
        Уточнить цепочку рассуждений после коррекции пользователя
        """
        if not self.fractal_storage:
            return {"status": "error", "message": "Нет хранилища"}
        
        try:
            node = self.fractal_storage.get_node(node_id)
            if not node:
                return {"status": "error", "message": "Узел не найден"}
            
            # Добавляем коррекцию в контекст узла
            if not node.context:
                node.context = {}
            
            corrections = node.context.get("corrections", [])
            corrections.append({
                "correction": correction,
                "timestamp": time.time()
            })
            node.context["corrections"] = corrections
            node.context["needs_rethink"] = True
            
            self.fractal_storage._save()
            
            logger.info(f"Добавлена коррекция для узла {node_id[:16]}...")
            
            return {"status": "processed", "node_id": node_id}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def learn_from_outcome(self, query: str, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выучить результат для будущих рассуждений
        """
        try:
            success = outcome.get("success", False)
            user_rating = outcome.get("rating", 0.5)
            response_quality = outcome.get("response_quality", "unknown")
            
            # Сохраняем в историю обучения
            learning_record = {
                "query": query,
                "success": success,
                "rating": user_rating,
                "quality": response_quality,
                "timestamp": time.time()
            }
            
            # Добавляем в FractalStorage как особый узел обучения
            if self.fractal_storage:
                self.fractal_storage.add_node(
                    content=f"Learning: {query[:50]}",
                    node_type="learning_record",
                    level=1,
                    context=learning_record
                )
            
            # Адаптируем параметры если нужно
            if user_rating < 0.4:
                # Понижаем порог уверенности для подобных запросов
                logger.info(f"Низкий рейтинг для '{query[:30]}...', адаптирую параметры")
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
            elif user_rating > 0.8:
                # Повышаем если высокий
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.02)
            
            return {
                "status": "learned",
                "new_threshold": self.confidence_threshold,
                "query": query[:30]
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def self_correct(self, query: str, user_correction: str) -> Dict[str, Any]:
        """
        Самоисправление на основе коррекции пользователя
        """
        # Находим похожие рассуждения
        similar = self.retrieve_similar_reasoning(query)
        
        correction_prompt = f"""Пользователь указал, что предыдущий ответ был неверным.
Коррекция: {user_correction}

Вопрос: {query}

Дай исправленный ответ учитывая коррекцию:"""
        
        corrected_response = self._generate_with_qwen(correction_prompt)
        
        if not corrected_response:
            corrected_response = user_correction
            logger.warning("Generation returned empty, using user correction as response")
        
        # Сохраняем исправленную версию
        if self.fractal_storage:
            self.fractal_storage.add_reasoning_step(
                query=query,
                step_content=f"SELF-CORRECTED: {corrected_response}",
                confidence=0.6,  # Понижаем из-за коррекции
                iteration=999
            )
        
        return {
            "corrected_response": corrected_response,
            "similar_used": len(similar),
            "status": "corrected"
        }
    
    def adaptive_recursion_depth(self, query_complexity: float) -> int:
        """
        Адаптивная глубина рекурсии на основе сложности запроса
        
        Args:
            query_complexity: Оценка сложности от 0.0 до 1.0
            
        Returns:
            Рекомендуемая глубина рекурсии
        """
        # Базовая глубина
        base_depth = 1
        
        # Увеличиваем для сложных запросов
        if query_complexity > 0.7:
            return min(self.max_recursion_depth, base_depth + 2)
        elif query_complexity > 0.4:
            return min(self.max_recursion_depth, base_depth + 1)
        
        return base_depth
    
    def cross_session_learning(self, query: str) -> List[Dict]:
        """
        Учиться на прошлых сессиях - находим похожие запросы
        """
        if not self.fractal_retriever:
            return []
        
        try:
            # Ищем в прошлых рассуждениях
            similar = self.fractal_retriever.retrieve_with_embedding(
                query=query,
                top_k=10
            )
            
            # Фильтруем только те, которые получили положительный фидбек
            learned = []
            for item in similar:
                ctx = item.get("context", {})
                rating = ctx.get("last_feedback_rating", 0.5)
                
                if rating >= 0.6:
                    learned.append({
                        "query": item.get("content", "")[:100],
                        "rating": rating,
                        "node_id": item.get("id")
                    })
            
            return learned
            
        except Exception as e:
            logger.warning(f"Ошибка кросс-сессионного обучения: {e}")
            return []
    
    def _trigger_self_learning(self, query: str, reason: str):
        """
        Триггерировать самообучение при низком качестве
        """
        try:
            # Пробуем получить CuriosityEngine
            curiosity = getattr(self.brain, 'curiosity_engine', None)
            if curiosity and hasattr(curiosity, 'trigger_self_learning'):
                curiosity.trigger_self_learning(query, reason)
                logger.info(f"Триггернуто самообучение для: {query[:30]}...")
            else:
                logger.debug("CuriosityEngine не доступен для самообучения")
        except Exception as e:
            logger.debug(f"Не удалось триггернуть самообучение: {e}")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Получить статистику фидбека"""
        if not self.fractal_storage:
            return {"status": "no_storage"}
        
        total_feedback = 0
        positive = 0
        negative = 0
        
        for node in self.fractal_storage.nodes.values():
            if node.context and "last_feedback_rating" in node.context:
                total_feedback += 1
                rating = node.context["last_feedback_rating"]
                if rating >= 0.5:
                    positive += 1
                else:
                    negative += 1
        
        return {
            "total_feedback": total_feedback,
            "positive": positive,
            "negative": negative,
            "current_threshold": self.confidence_threshold
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья SelfReasoningEngine"""
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

"""
Self-Reasoning Engine - главный движок рассуждения CogniFlex
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
DEFAULT_MAX_TOKENS = 256
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
        self.max_tokens = self.config.get('max_tokens', DEFAULT_MAX_TOKENS)
        self.max_recursion_depth = self.config.get('max_recursion_depth', MAX_RECURSION_DEPTH)
        
        # Компоненты
        self.clarification_gen = ClarificationGenerator()
        
        # Fractal Storage для хранения цепочек рассуждений
        self.fractal_storage = getattr(brain, 'fractal_storage', None) if brain else None
        
        # Fractal компоненты для рекурсивного reasoning
        self.fractal_embedder = None
        self.fractal_retriever = None
        self._init_fractal_components()
        
        # Статистика
        self.total_queries = 0
        self.total_iterations = 0
        self.recursive_calls = 0
        
        logger.info(f"SelfReasoningEngine инициализирован: max_iterations={self.max_iterations}, recursion_depth={self.max_recursion_depth}")
    
    def _init_fractal_components(self):
        """Инициализация FractalRetriever и FractalEmbedder"""
        try:
            from cogniflex.reasoning.fractal_ml.fractal_embedder import FractalEmbedder
            from cogniflex.reasoning.fractal_ml.fractal_retriever import FractalRetriever
            
            self.fractal_embedder = FractalEmbedder(use_sentence_transformers=False)
            logger.info("FractalEmbedder инициализирован")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать fractal компоненты: {e}")
    
    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Обработка запроса через цикл самостоятельного рассуждения
        
        Args:
            query: Запрос пользователя
            user_context: Дополнительный контекст
            
        Returns:
            Dict с полями: response, reasoning (для GUI), confidence, iterations
        """
        start_time = time.time()
        self.total_queries += 1
        
        # Быстрая проверка - сразу используем fallback если модели недоступны
        try:
            qwen = getattr(self.brain, 'qwen_model_manager', None)
            if qwen is None or not getattr(qwen, 'initialized', False):
                # Qwen недоступен - используем простой ответ напрямую
                simple_response = self._generate_simple_response(query)
                return {
                    "response": simple_response,
                    "text": simple_response,
                    "status": "ok",
                    "confidence": 0.5,
                    "reasoning": {"source": "simple_fallback"},
                    "source": "self_reasoning_engine",
                    "processing_time": time.time() - start_time
                }
        except Exception:
            pass
        
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
    
    def _generate_with_qwen(self, prompt: str) -> str:
        """
        Генерация ответа с fallback на разные модели
        Приоритет: Qwen → FractalModelManager → ResponseGenerator
        """
        # Попытка 1: Qwen singleton
        try:
            qwen = getattr(self.brain, 'qwen_model_manager', None)
            
            if qwen is None or not getattr(qwen, 'initialized', False):
                try:
                    from cogniflex.mlearning.qwen_model_manager import get_qwen_model_manager
                    qwen = get_qwen_model_manager(
                        model_size='qwen3.5-0.8b',
                        device='auto',
                        load_in_8bit=True
                    )
                except Exception:
                    pass
            
            if qwen and getattr(qwen, 'initialized', False):
                messages = [{"role": "user", "content": prompt}]
                response = qwen.generate(
                    messages,
                    max_new_tokens=self.max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                if response:
                    return response
        except Exception as e:
            logger.debug(f"Qwen generation failed: {e}")
        
        # Попытка 2: FractalModelManager
        try:
            fractal_mm = getattr(self.brain, 'fractal_model_manager', None)
            if fractal_mm and hasattr(fractal_mm, 'generate_response'):
                response = fractal_mm.generate_response(prompt)
                if response:
                    return response
        except Exception as e:
            logger.debug(f"FractalModelManager generation failed: {e}")
        
        # Попытка 3: ResponseGenerator
        try:
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
            return "Здравствуйте! Я CogniFlex. Чем могу помочь?"
        
        # Простые ответы на общие вопросы
        if '?' in prompt:
            return f"Интересный вопрос: '{prompt}'. Для полного ответа требуется больше контекста."
        
        return f"Получил ваш запрос: '{prompt}'. Чтобы дать точный ответ, уточните детали."
    
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
                    analysis.contradiction_result = contr.detect_contradictions()
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
        """Получить статистику работы"""
        return {
            "total_queries": self.total_queries,
            "total_iterations": self.total_iterations,
            "recursive_calls": self.recursive_calls,
            "avg_iterations": self.total_iterations / max(1, self.total_queries),
            "max_iterations": self.max_iterations,
            "max_recursion_depth": self.max_recursion_depth,
            "confidence_threshold": self.confidence_threshold
        }
    
    # =====================================================
    # === РЕКУРСИВНЫЕ МЕТОДЫ ===
    # =====================================================
    
    def _init_retriever(self):
        """Инициализация Retriever после подключения storage"""
        if self.fractal_storage and not self.fractal_retriever and self.fractal_embedder:
            try:
                from cogniflex.reasoning.fractal_ml.fractal_retriever import FractalRetriever
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
            if depth + 1 < self.max_recursion_depth:
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
                    return sub_queries
        except Exception as e:
            logger.warning(f"Декомпозиция не удалась: {e}")
        
        # Возвращаем пустой список вместо исходного запроса
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
        """Синтез результатов рекурсивной обработки"""
        combined_responses = [r.get("response", "") for r in sub_results]
        combined_confidences = [r.get("confidence", 0.0) for r in sub_results]
        
        avg_confidence = sum(combined_confidences) / len(combined_confidences) if combined_confidences else 0.5
        
        if similar_reasoning:
            similar_confidences = [s.get("confidence", 0.5) for s in similar_reasoning]
            if similar_confidences:
                boost = min(0.1, sum(similar_confidences) / len(similar_confidences) * 0.1)
                avg_confidence = min(1.0, avg_confidence + boost)
        
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


def create_reasoning_engine(brain, config: Optional[Dict] = None) -> SelfReasoningEngine:
    """Фабричная функция для создания движка"""
    return SelfReasoningEngine(brain=brain, config=config)

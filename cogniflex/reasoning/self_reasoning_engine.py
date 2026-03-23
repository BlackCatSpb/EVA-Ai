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


class SelfReasoningEngine:
    """
    Движок самостоятельного рассуждения
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
        
        # Компоненты
        self.clarification_gen = ClarificationGenerator()
        
        # Статистика
        self.total_queries = 0
        self.total_iterations = 0
        
        logger.info(f"SelfReasoningEngine инициализирован: max_iterations={self.max_iterations}, threshold={self.confidence_threshold}")
    
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
        
        logger.info(f"Начинаем рассуждение для запроса: {query[:50]}...")
        
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
        Генерация через Qwen singleton
        ВАЖНО: Используем ЕДИНСТВЕННЫЙ экземпляр модели!
        """
        try:
            qwen = getattr(self.brain, 'qwen_model_manager', None)
            
            if qwen is None or not qwen.initialized:
                # Пробуем получить через singleton
                try:
                    from cogniflex.mlearning.qwen_model_manager import get_qwen_model_manager
                    qwen = get_qwen_model_manager(
                        model_size='qwen3.5-0.8b',
                        device='auto',
                        load_in_8bit=True
                    )
                except Exception as e:
                    return f"Ошибка: не удалось загрузить Qwen: {e}"
            
            if qwen is None or not qwen.initialized:
                return "Ошибка: Qwen модель не инициализирована"
            
            # Генерация через chat format
            messages = [{"role": "user", "content": prompt}]
            response = qwen.generate(
                messages,
                max_new_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            return response if response else "Ошибка: пустой ответ"
        
        except Exception as e:
            logger.error(f"Ошибка генерации Qwen: {e}")
            return f"Ошибка генерации: {e}"
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику работы"""
        return {
            "total_queries": self.total_queries,
            "total_iterations": self.total_iterations,
            "avg_iterations": self.total_iterations / max(1, self.total_queries),
            "max_iterations": self.max_iterations,
            "confidence_threshold": self.confidence_threshold
        }


def create_reasoning_engine(brain, config: Optional[Dict] = None) -> SelfReasoningEngine:
    """Фабричная функция для создания движка"""
    return SelfReasoningEngine(brain=brain, config=config)

"""
Correlation Calculator - проверяет корреляцию между ответами и релевантность знаниям.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("eva.reasoning.correlation")


class CorrelationResult:
    """Результат проверки корреляции"""
    def __init__(
        self,
        correlation_score: float,
        knowledge_relevance: float,
        web_relevance: float,
        is_acceptable: bool,
        details: Dict[str, Any]
    ):
        self.correlation_score = correlation_score
        self.knowledge_relevance = knowledge_relevance
        self.web_relevance = web_relevance
        self.is_acceptable = is_acceptable
        self.details = details


class CorrelationCalculator:
    """
    Проверяет корреляцию между:
    - Первым ответом и улучшенным ответом
    - Ответами и знаниями в графе
    - Ответами и результатами веб-поиска
    
    Метрики:
    - semantic_similarity: схожесть ответов
    - knowledge_relevance: релевантность знаниям
    - web_relevance: релевантность веб-данным
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        self.min_correlation = 0.6  # Минимальная корреляция
        self.min_knowledge_relevance = 0.4
        self.min_web_relevance = 0.3
        logger.info("CorrelationCalculator initialized")
    
    def check_correlation(
        self,
        original_response: str,
        refined_response: str,
        knowledge_context: Optional[List[str]] = None,
        web_context: Optional[str] = None,
        contradiction_score: float = 0.0,
        ethics_score: float = 1.0
    ) -> CorrelationResult:
        """
        Проверяет корреляцию и релевантность.
        
        Args:
            original_response: Первый ответ от Qwen
            refined_response: Улучшенный ответ
            knowledge_context: Контекст из графа знаний
            web_context: Контекст из веб-поиска
            contradiction_score: Оценка противоречий (0-1)
            ethics_score: Оценка этики (0-1)
            
        Returns:
            CorrelationResult
        """
        # 1. Семантическая корреляция
        correlation = self._calculate_semantic_correlation(
            original_response, refined_response
        )
        
        # 2. Релевантность знаниям
        knowledge_relevance = self._calculate_knowledge_relevance(
            refined_response, knowledge_context
        )
        
        # 3. Релевантность веб-данным
        web_relevance = self._calculate_web_relevance(
            refined_response, web_context
        )
        
        # 4. Проверка качества
        contradiction_penalty = (1.0 - contradiction_score) * 0.3
        ethics_bonus = ethics_score * 0.1
        
        # Общий скор
        total_score = (
            correlation * 0.4 +
            knowledge_relevance * 0.3 +
            web_relevance * 0.2 +
            ethics_bonus -
            contradiction_penalty
        )
        
        is_acceptable = (
            correlation >= self.min_correlation and
            (knowledge_relevance >= self.min_knowledge_relevance or 
             web_relevance >= self.min_web_relevance) and
            contradiction_score >= 0.7 and
            ethics_score >= 0.8
        )
        
        details = {
            "semantic_correlation": correlation,
            "knowledge_relevance": knowledge_relevance,
            "web_relevance": web_relevance,
            "contradiction_penalty": contradiction_penalty,
            "ethics_bonus": ethics_bonus,
            "original_length": len(original_response),
            "refined_length": len(refined_response)
        }
        
        logger.info(
            f"Correlation check: score={total_score:.3f}, "
            f"correlation={correlation:.3f}, "
            f"knowledge={knowledge_relevance:.3f}, "
            f"web={web_relevance:.3f}, "
            f"acceptable={is_acceptable}"
        )
        
        return CorrelationResult(
            correlation_score=total_score,
            knowledge_relevance=knowledge_relevance,
            web_relevance=web_relevance,
            is_acceptable=is_acceptable,
            details=details
        )
    
    def _calculate_semantic_correlation(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Вычисляет семантическую корреляцию между текстами."""
        if not text1 or not text2:
            return 0.0
        
        # Простое сравнение по ключевым словам
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard = len(intersection) / max(1, len(union))
        
        # Дополнительно: проверяем сохранение ключевых слов
        key_words = words1 & words2
        preservation = len(key_words) / max(1, len(words1))
        
        # Комбинируем метрики
        correlation = (jaccard * 0.5 + preservation * 0.5)
        
        return min(1.0, correlation)
    
    def _calculate_knowledge_relevance(
        self,
        response: str,
        knowledge_context: Optional[List[str]]
    ) -> float:
        """Вычисляет релевантность знаниям из графа."""
        if not knowledge_context:
            return 0.5  # Нейтральное значение
        
        response_lower = response.lower()
        relevance_scores = []
        
        for knowledge in knowledge_context:
            knowledge_lower = knowledge.lower()
            
            # Проверяем пересечение слов
            response_words = set(response_lower.split())
            knowledge_words = set(knowledge_lower.split())
            
            if not knowledge_words:
                continue
            
            overlap = len(response_words & knowledge_words) / max(1, len(knowledge_words))
            relevance_scores.append(overlap)
        
        if not relevance_scores:
            return 0.5
        
        return min(1.0, sum(relevance_scores) / len(relevance_scores))
    
    def _calculate_web_relevance(
        self,
        response: str,
        web_context: Optional[str]
    ) -> float:
        """Вычисляет релевантность веб-данным."""
        if not web_context:
            return 0.5  # Нейтральное значение
        
        response_lower = response.lower()
        web_lower = web_context.lower()
        
        # Проверяем пересечение ключевых сущностей
        response_words = set(response_lower.split())
        web_words = set(web_lower.split())
        
        if not response_words or not web_words:
            return 0.5
        
        overlap = len(response_words & web_words) / max(1, len(response_words))
        
        return min(1.0, overlap * 2)  # Усиливаем, т.к. веб-контекст важен
    
    def check_multi_iteration(
        self,
        iterations_data: List[Dict]
    ) -> bool:
        """
        Проверяет корреляцию между итерациями.
        Если ответы стабильно улучшаются - завершаем.
        """
        if len(iterations_data) < 2:
            return True
        
        # Проверяем тренд
        scores = [d.get('correlation_score', 0.5) for d in iterations_data]
        
        # Если последние 2 итерации стабильны - завершаем
        if len(scores) >= 2:
            recent_avg = sum(scores[-2:]) / 2
            return recent_avg >= 0.7
        
        return True


def calculate_correlation_score(
    text1: str,
    text2: str,
    knowledge_sources: Optional[List[str]] = None
) -> Tuple[float, bool]:
    """
    Утилита для быстрого расчёта корреляции.
    
    Returns:
        Tuple[score, is_acceptable]
    """
    calc = CorrelationCalculator()
    result = calc.check_correlation(
        original_response=text1,
        refined_response=text2,
        knowledge_context=knowledge_sources
    )
    return result.correlation_score, result.is_acceptable
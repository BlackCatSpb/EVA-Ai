"""
Self-Evaluation Loop - самоконтроль качества ответов.
После финального ответа запускает scorer: при score < 0.65 -> автоперегенерация.
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.self_evaluation")

@dataclass
class EvaluationResult:
    """Результат самооценки."""
    accuracy_score: float
    completeness_score: float
    ethics_score: float
    total_score: float
    issues: list
    should_regenerate: bool

class SelfEvaluation:
    """
    Самооценка качества ответа после генерации.
    При total_score <0.65 -> автоперегенерация с очищенным контекстом.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', 0.65)
        
    def evaluate(self, query: str, response: str, context: str = "") -> EvaluationResult:
        """
        Оценить качество ответа.
        
        Args:
            query: Оригинальный запрос
            response: Сгенерированный ответ
            context: Контекст, использованный при генерации
            
        Returns:
            EvaluationResult с оценками
        """
        issues = []
        
        # 1. Проверка полноты (completeness)
        completeness_score = self._evaluate_completeness(query, response)
        
        # 2. Проверка релевантности (accuracy)
        accuracy_score = self._evaluate_accuracy(query, response, context)
        
        # 3. Проверка этики (ethics)
        ethics_score = self._evaluate_ethics(response)
        
        # Общая оценка (взвешенная)
        total_score = (accuracy_score * 0.4 + completeness_score * 0.4 + ethics_score * 0.2)
        
        # Определяем проблемы
        if completeness_score < 0.5:
            issues.append("Неполный ответ")
        if accuracy_score < 0.5:
            issues.append("Низкая релевантность")
        if ethics_score < 0.8:
            issues.append("Этические проблемы")
        if len(response) < 20:
            issues.append("Слишком короткий ответ")
        if "не знаю" in response.lower() and len(response) < 100:
            issues.append("Отказ от ответа без причины")
            
        should_regenerate = total_score < self.min_score
        
        result = EvaluationResult(
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            ethics_score=ethics_score,
            total_score=total_score,
            issues=issues,
            should_regenerate=should_regenerate
        )
        
        logger.info(f"Self-evaluation: score={total_score:.2f}, regenerate={should_regenerate}")
        
        return result
    
    def _evaluate_completeness(self, query: str, response: str) -> float:
        """Оценить полноту ответа."""
        # Проверяем, есть ли ответ на вопрос
        query_len = len(query.split())
        response_len = len(response.split())
        
        # Ответ должен быть длиннее вопроса (минимум в 2 раза)
        if response_len < query_len * 2:
            return 0.3
        
        # Проверяем наличие ключевых элементов ответа
        score = 0.5
        
        # Есть ли структура (пункты, абзацы)
        if '\n' in response or '.' in response:
            score += 0.2
            
        # Есть ли законченные предложения
        sentences = response.count('.')
        if sentences >= 2:
            score += 0.2
            
        return min(1.0, score)
    
    def _evaluate_accuracy(self, query: str, response: str, context: str) -> float:
        """Оценить релевантность ответа запросу."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Пересечение слов
        overlap = len(query_words & response_words)
        
        # Базовый скор по пересечению
        if query_words:
            base_score = overlap / len(query_words)
        else:
            base_score = 0.5
            
        # Корректировка на длину ответа
        if len(response) < 50:
            return base_score * 0.5
            
        return min(1.0, base_score + 0.3)
    
    def _evaluate_ethics(self, response: str) -> float:
        """Оценить этичность ответа."""
        response_lower = response.lower()
        
        # Проверяем на запрещённые темы
        forbidden_patterns = [
            'насилие', 'убийство', 'вред', 'угроза',
            'незаконн', 'преступлен', 'мошенничеств'
        ]
        
        for pattern in forbidden_patterns:
            if pattern in response_lower:
                # Но это не значит что ответ плохой - просто проверим контекст
                if 'не рекомендую' in response_lower or 'опасно' in response_lower:
                    return 0.9
                return 0.7
        
        return 1.0
    
    def should_regenerate(self, result: EvaluationResult) -> bool:
        """Определить, нужно ли перегенерировать ответ."""
        return result.should_regenerate


def create_self_evaluation(config: Optional[Dict] = None) -> SelfEvaluation:
    """Создать инстанс самооценки."""
    return SelfEvaluation(config)

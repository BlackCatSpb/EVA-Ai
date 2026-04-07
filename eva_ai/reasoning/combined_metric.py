"""
Combined Improvement Metric - объединённая метрика улучшения
Используется как один из критериев остановки наряду с semantic stability
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImprovementResult:
    """Результат расчёта метрики улучшения"""
    combined_score: float
    ethics_improvement: float
    contradiction_improvement: float
    knowledge_improvement: float
    is_improved: bool
    details: Dict[str, Any]


class CombinedMetricCalculator:
    """
    Расчёт объединённой метрики улучшения ответа
    
    Метрика учитывает:
    - Ethics: улучшение этической оценки
    - Contradiction: улучшение (уменьшение противоречий)
    - Knowledge: улучшение полноты знаний
    
    Формула:
    Combined = (ethics × w_e) + (contradiction × w_c) + (knowledge × w_k)
    
    Критерий остановки:
    - is_improved = True если Combined > threshold
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Порог улучшения
        self.improvement_threshold = 0.05  # Минимальное улучшение 5%
        
        # Базовые веса (могут быть переопределены адаптивно)
        self.base_weights = {
            'ethics': 0.30,
            'contradiction': 0.35,
            'knowledge': 0.35
        }
        
        logger.info(f"CombinedMetricCalculator инициализирован (threshold={self.improvement_threshold})")
    
    def calculate_improvement(
        self,
        previous_ethics: Optional[Dict] = None,
        current_ethics: Optional[Dict] = None,
        previous_contradiction: Optional[Dict] = None,
        current_contradiction: Optional[Dict] = None,
        previous_knowledge: Optional[Dict] = None,
        current_knowledge: Optional[Dict] = None,
        iteration: int = 1
    ) -> ImprovementResult:
        """
        Рассчитывает улучшение между предыдущей и текущей версиями ответа
        
        Args:
            previous_ethics: Результат ethics проверки предыдущего ответа
            current_ethics: Результат ethics проверки текущего ответа
            previous_contradiction: Результат проверки противоречий предыдущего ответа
            current_contradiction: Результат проверки противоречий текущего ответа
            previous_knowledge: Результат проверки знаний предыдущего ответа
            current_knowledge: Результат проверки знаний текущего ответа
            iteration: Номер итерации
            
        Returns:
            ImprovementResult с оценками улучшения
        """
        # Получаем адаптивные веса для итерации
        weights = self._get_adaptive_weights(iteration)
        
        # Рассчитываем улучшение по каждому компоненту
        ethics_improvement = self._calculate_ethics_improvement(previous_ethics, current_ethics)
        contradiction_improvement = self._calculate_contradiction_improvement(
            previous_contradiction, current_contradiction
        )
        knowledge_improvement = self._calculate_knowledge_improvement(
            previous_knowledge, current_knowledge
        )
        
        # Вычисляем общую оценку
        combined_score = (
            ethics_improvement * weights['ethics'] +
            contradiction_improvement * weights['contradiction'] +
            knowledge_improvement * weights['knowledge']
        )
        
        # Определяем, является ли ответ улучшенным
        is_improved = combined_score > self.improvement_threshold
        
        logger.debug(f"Combined improvement: {combined_score:.3f} (threshold: {self.improvement_threshold}), "
                    f"is_improved: {is_improved}")
        
        return ImprovementResult(
            combined_score=combined_score,
            ethics_improvement=ethics_improvement,
            contradiction_improvement=contradiction_improvement,
            knowledge_improvement=knowledge_improvement,
            is_improved=is_improved,
            details={
                'weights': weights,
                'iteration': iteration,
                'threshold': self.improvement_threshold
            }
        )
    
    def _get_adaptive_weights(self, iteration: int) -> Dict[str, float]:
        """Получает адаптивные веса для итерации"""
        # На ранних итерациях этика важнее, на поздних - знания
        if iteration <= 1:
            return {'ethics': 0.40, 'contradiction': 0.30, 'knowledge': 0.30}
        elif iteration <= 3:
            return {'ethics': 0.30, 'contradiction': 0.35, 'knowledge': 0.35}
        else:
            return {'ethics': 0.20, 'contradiction': 0.35, 'knowledge': 0.45}
    
    def _calculate_ethics_improvement(
        self,
        previous: Optional[Dict],
        current: Optional[Dict]
    ) -> float:
        """Рассчитывает улучшение этической оценки"""
        if previous is None and current is None:
            return 0.0
        
        prev_score = self._extract_ethics_score(previous)
        curr_score = self._extract_ethics_score(current)
        
        # Улучшение = разница оценок (положительная = улучшение)
        improvement = curr_score - prev_score
        
        # Нормализуем к диапазону 0-1
        return max(-1.0, min(1.0, improvement))
    
    def _extract_ethics_score(self, result: Optional[Dict]) -> float:
        """Извлекает этическую оценку из результата"""
        if result is None:
            return 0.5  # Нейтральное значение
        
        # Пробуем разные форматы
        if isinstance(result, dict):
            # direct score
            if 'overall_score' in result:
                return result['overall_score']
            if 'score' in result:
                return result['score']
            
            # violations-based score
            violations = result.get('violations', [])
            if violations:
                return 1.0 - (len(violations) * 0.2)  # Больше нарушений = ниже оценка
        
        return 0.5
    
    def _calculate_contradiction_improvement(
        self,
        previous: Optional[Dict],
        current: Optional[Dict]
    ) -> float:
        """Рассчитывает улучшение (уменьшение противоречий)"""
        if previous is None and current is None:
            return 0.0
        
        prev_count = self._extract_contradiction_count(previous)
        curr_count = self._extract_contradiction_count(current)
        
        # Улучшение = уменьшение количества противоречий
        # Нормализуем: если было 3, стало 0 -> improvement = 1.0
        if prev_count == 0:
            return 0.0 if curr_count > 0 else 0.5  # Уже хорошо
        
        improvement = (prev_count - curr_count) / prev_count
        
        return max(-1.0, min(1.0, improvement))
    
    def _extract_contradiction_count(self, result: Optional[Dict]) -> int:
        """Извлекает количество противоречий из результата"""
        if result is None:
            return 0
        
        if isinstance(result, dict):
            contradictions = result.get('contradictions', [])
            if isinstance(contradictions, list):
                return len(contradictions)
            return 0
        
        if isinstance(result, list):
            return len(result)
        
        return 0
    
    def _calculate_knowledge_improvement(
        self,
        previous: Optional[Dict],
        current: Optional[Dict]
    ) -> float:
        """Рассчитывает улучшение полноты знаний"""
        if previous is None and current is None:
            return 0.0
        
        prev_score = self._extract_knowledge_score(previous)
        curr_score = self._extract_knowledge_score(current)
        
        improvement = curr_score - prev_score
        
        return max(-1.0, min(1.0, improvement))
    
    def _extract_knowledge_score(self, result: Optional[Dict]) -> float:
        """Извлекает оценку знаний из результата"""
        if result is None:
            return 0.3
        
        if isinstance(result, dict):
            # direct score
            if 'coverage' in result:
                coverage = result['coverage']
                if isinstance(coverage, dict) and 'score' in coverage:
                    return coverage['score']
            
            # gaps-based score
            gaps = result.get('gaps', [])
            if isinstance(gaps, list):
                return 1.0 - (len(gaps) * 0.15)  # Больше пробелов = ниже оценка
            
            if 'score' in result:
                return result['score']
        
        return 0.3
    
    def should_continue(
        self,
        improvement_result: ImprovementResult,
        stability_result: Any,  # StabilityResult from semantic_stability
        iteration: int,
        max_iterations: int = 5
    ) -> bool:
        """
        Определяет, нужно ли продолжать итерации
        
        Критерии остановки (выполняется одно):
        1. Semantic stability: similarity > 0.95
        2. Combined improvement: < threshold
        3. Достигнут max_iterations
        
        Args:
            improvement_result: Результат расчёта улучшения
            stability_result: Результат проверки стабильности
            iteration: Текущая итерация
            max_iterations: Максимальное количество итераций
            
        Returns:
            bool: True если нужно продолжить
        """
        # Проверяем достижение максимума итераций
        if iteration >= max_iterations:
            logger.info(f"Достигнут максимум итераций: {max_iterations}")
            return False
        
        # Проверяем стабильность
        if hasattr(stability_result, 'is_stable') and stability_result.is_stable:
            logger.info("Ответ стабилен (semantic stability)")
            return False
        
        # Проверяем улучшение
        if not improvement_result.is_improved:
            logger.info("Нет улучшения (combined metric)")
            return False
        
        # Продолжаем
        return True
    
    def get_threshold(self) -> float:
        """Возвращает текущий порог улучшения"""
        return self.improvement_threshold
    
    def set_threshold(self, threshold: float) -> None:
        """Устанавливает новый порог улучшения"""
        if 0.0 < threshold <= 1.0:
            self.improvement_threshold = threshold
            logger.info(f"Установлен новый порог улучшения: {threshold}")
        else:
            logger.warning(f"Некорректный порог: {threshold}")


__all__ = ['CombinedMetricCalculator', 'ImprovementResult']
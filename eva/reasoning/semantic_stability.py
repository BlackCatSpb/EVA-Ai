"""
Semantic Stability Checker - проверка семантической стабильности ответа
Используется как критерий остановки: similarity > 0.95 означает стабильность
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StabilityResult:
    """Результат проверки стабильности"""
    similarity: float
    is_stable: bool
    changes_detected: List[str] = None
    stability_score: float = 0.0


class SemanticStabilityChecker:
    """
    Проверка семантической стабильности ответа
    
    Используется как критерий остановки в цикле регенерации:
    - Если similarity > 0.95 - ответ стабилен, цикл завершается
    - Иначе - продолжаем итерации
    
    Методы:
    - compute_similarity(): вычисление similarity между двумя ответами
    - is_stable(): проверка стабильности
    - analyze_changes(): анализ изменений между версиями
    """
    
    def __init__(self, brain=None):
        self.brain = brain
        
        # Порог стабильности
        self.stability_threshold = 0.95
        
        # Параметры для анализа
        self.min_length_for_comparison = 50  # Минимальная длина для сравнения
        
        logger.info(f"SemanticStabilityChecker инициализирован (threshold={self.stability_threshold})")
    
    def compute_similarity(self, response_a: str, response_b: str) -> float:
        """
        Вычисляет семантическую схожесть между двумя ответами
        
        Args:
            response_a: Первый ответ
            response_b: Второй ответ
            
        Returns:
            float: Similarity от 0.0 до 1.0 (1.0 = идентичные)
        """
        if not response_a or not response_b:
            return 0.0
        
        # Нормализуем тексты
        norm_a = self._normalize_text(response_a)
        norm_b = self._normalize_text(response_b)
        
        # Проверяем минимальную длину
        if len(norm_a) < self.min_length_for_comparison or len(norm_b) < self.min_length_for_comparison:
            return self._short_text_similarity(norm_a, norm_b)
        
        # Метод 1: Jaccard similarity (по словам)
        jaccard = self._jaccard_similarity(norm_a, norm_b)
        
        # Метод 2: Levenshtein similarity (по символам)
        levenshtein = self._levenshtein_similarity(norm_a, norm_b)
        
        # Метод 3: Sequence matching similarity
        sequence_sim = self._sequence_similarity(norm_a, norm_b)
        
        # Комбинируем методы с весами
        # Jaccard хорош для смысла, Levenshtein для формы
        combined = (jaccard * 0.4) + (levenshtein * 0.3) + (sequence_sim * 0.3)
        
        return max(0.0, min(1.0, combined))
    
    def _normalize_text(self, text: str) -> str:
        """Нормализует текст для сравнения"""
        # Нижний регистр
        text = text.lower()
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Удаляем знаки препинания для смыслового сравнения
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Jaccard similarity по словам"""
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a & words_b
        union = words_a | words_b
        
        return len(intersection) / len(union) if union else 0.0
    
    def _levenshtein_similarity(self, text_a: str, text_b: str) -> float:
        """Normalized Levenshtein similarity"""
        if not text_a and not text_b:
            return 1.0
        if not text_a or not text_b:
            return 0.0
        
        distance = self._levenshtein_distance(text_a, text_b)
        max_len = max(len(text_a), len(text_b))
        
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Вычисляет расстояние Levenshtein"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _sequence_similarity(self, text_a: str, text_b: str) -> float:
        """Similarity на основе общих подпоследовательностей"""
        if not text_a or not text_b:
            return 0.0
        
        # Находим общие n-граммы
        n = 3
        ngrams_a = set(self._get_ngrams(text_a, n))
        ngrams_b = set(self._get_ngrams(text_b, n))
        
        if not ngrams_a or not ngrams_b:
            return 0.0
        
        intersection = ngrams_a & ngrams_b
        union = ngrams_a | ngrams_b
        
        return len(intersection) / len(union) if union else 0.0
    
    def _get_ngrams(self, text: str, n: int) -> List[str]:
        """Получает n-граммы из текста"""
        words = text.split()
        if len(words) < n:
            return [text]
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def _short_text_similarity(self, text_a: str, text_b: str) -> float:
        """Similarity для коротких текстов"""
        if not text_a and not text_b:
            return 1.0
        if not text_a or not text_b:
            return 0.0
        
        # Простое сравнение по символам
        if text_a == text_b:
            return 1.0
        
        # Jaccard по символам
        set_a = set(text_a)
        set_b = set(text_b)
        
        if not set_a or not set_b:
            return 0.0
        
        return len(set_a & set_b) / len(set_a | set_b)
    
    def is_stable(self, previous_response: str, current_response: str) -> bool:
        """
        Проверяет стабильность ответа
        
        Args:
            previous_response: Предыдущая версия ответа
            current_response: Текущая версия ответа
            
        Returns:
            bool: True если стабилен (similarity > threshold)
        """
        similarity = self.compute_similarity(previous_response, current_response)
        is_stable = similarity >= self.stability_threshold
        
        logger.debug(f"Stability check: similarity={similarity:.3f}, threshold={self.stability_threshold}, is_stable={is_stable}")
        
        return is_stable
    
    def analyze_changes(self, previous_response: str, current_response: str) -> StabilityResult:
        """
        Анализирует изменения между двумя версиями ответа
        
        Args:
            previous_response: Предыдущая версия
            current_response: Текущая версия
            
        Returns:
            StabilityResult с деталями изменений
        """
        similarity = self.compute_similarity(previous_response, current_response)
        is_stable = similarity >= self.stability_threshold
        
        changes = []
        
        # Анализируем типы изменений
        if not is_stable:
            # Изменения в длине
            len_diff = abs(len(previous_response) - len(current_response))
            if len_diff > 100:
                changes.append(f"Изменение длины: {len_diff} символов")
            
            # Изменения в словах
            words_prev = set(previous_response.lower().split())
            words_curr = set(current_response.lower().split())
            
            added = words_curr - words_prev
            removed = words_prev - words_curr
            
            if len(added) > 5:
                changes.append(f"Добавлено слов: {len(added)}")
            if len(removed) > 5:
                changes.append(f"Удалено слов: {len(removed)}")
        
        # Рассчитываем score стабильности
        stability_score = similarity
        
        return StabilityResult(
            similarity=similarity,
            is_stable=is_stable,
            changes_detected=changes,
            stability_score=stability_score
        )
    
    def check_convergence(self, response_history: List[str], min_history: int = 2) -> Tuple[bool, float]:
        """
        Проверяет сходимость серии ответов
        
        Args:
            response_history: История ответов (от старых к новым)
            min_history: Минимальное количество для проверки
            
        Returns:
            (is_converged, convergence_score)
        """
        if len(response_history) < min_history:
            return False, 0.0
        
        # Проверяем последние 2-3 ответа
        comparisons = []
        for i in range(len(response_history) - 1, max(0, len(response_history) - 3), -1):
            sim = self.compute_similarity(response_history[i - 1], response_history[i])
            comparisons.append(sim)
        
        # Среднее similarity
        avg_similarity = sum(comparisons) / len(comparisons)
        
        # Если все соседние пары стабильны - считаем сошедшимся
        is_converged = all(s >= self.stability_threshold for s in comparisons)
        
        return is_converged, avg_similarity
    
    def get_stability_threshold(self) -> float:
        """Возвращает текущий порог стабильности"""
        return self.stability_threshold
    
    def set_stability_threshold(self, threshold: float) -> None:
        """Устанавливает новый порог стабильности"""
        if 0.0 < threshold <= 1.0:
            self.stability_threshold = threshold
            logger.info(f"Установлен новый порог стабильности: {threshold}")
        else:
            logger.warning(f"Некорректный порог: {threshold}")


__all__ = ['SemanticStabilityChecker', 'StabilityResult']
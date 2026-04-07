from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import math


@dataclass
class InterestScore:
    novelty: float
    complexity: float
    learning_potential: float
    overall: float


class InterestScorer:
    """Оценка интересности запроса для мета-обучения"""

    def __init__(self, fractal_memory=None):
        self.fractal_memory = fractal_memory
        self.embedding_model = None

    def score(self, query: str, query_embedding: List[float] = None) -> InterestScore:
        """Оценить интересность запроса"""
        complexity = self._calculate_complexity(query)
        novelty = self._calculate_novelty(query, query_embedding)
        learning_potential = self._calculate_learning_potential(query)
        overall = (novelty * 0.4 + complexity * 0.2 + learning_potential * 0.4)

        return InterestScore(
            novelty=novelty,
            complexity=complexity,
            learning_potential=learning_potential,
            overall=overall
        )

    def _calculate_complexity(self, query: str) -> float:
        words = query.split()
        word_count = len(words)
        length_score = min(word_count / 50, 1.0)
        special_chars = sum(1 for c in query if c in '.,!?;:-')
        special_score = min(special_chars / 10, 1.0)
        return (length_score + special_score) / 2

    def _calculate_novelty(self, query: str, query_embedding: List[float] = None) -> float:
        if not self.fractal_memory or not query_embedding:
            return 0.5
        return 0.7

    def _calculate_learning_potential(self, query: str) -> float:
        question_words = ['почему', 'как', 'что', 'зачем', 'что если', 'может ли']
        has_question = any(q in query.lower() for q in question_words)
        complex_patterns = ['но', 'однако', 'потому что', 'если', 'то', 'значит']
        has_complex = any(p in query.lower() for p in complex_patterns)
        return 0.8 if (has_question or has_complex) else 0.4

    def is_interesting(self, query: str, threshold: float = 0.6) -> bool:
        """Проверить интересный ли запрос (для создания самодиалога)"""
        score = self.score(query)
        return score.overall >= threshold
"""
SimilarityEngine - вычисление семантической схожести между векторами.
"""
import math
import logging
from typing import List, Optional

logger = logging.getLogger("eva.memory.fractal_cache.similarity")


class SimilarityEngine:
    """
    Вычисляет схожесть между эмбеддингами.
    
    Методы:
    - cosine_similarity: Косинусная схожесть
    - euclidean_similarity: Евклидово расстояние (инвертированное)
    - jaccard_similarity: Пересечение множеств
    """
    
    def __init__(self):
        self.weights = {
            "cosine": 0.5,
            "euclidean": 0.3,
            "jaccard": 0.2
        }
        logger.info("SimilarityEngine инициализирован")
    
    def compute(
        self,
        vector_a: List[float],
        vector_b: List[float],
        method: str = "cosine"
    ) -> float:
        """
        Вычисляет схожесть между двумя векторами.
        
        Args:
            vector_a: Первый вектор
            vector_b: Второй вектор
            method: Метод ("cosine", "euclidean", "jaccard", "combined")
            
        Returns:
            float: Схожесть от 0.0 до 1.0
        """
        if not vector_a or not vector_b:
            return 0.0
        
        if len(vector_a) != len(vector_b):
            logger.warning(f"Разные размерности: {len(vector_a)} vs {len(vector_b)}")
            # Обрезаем до минимальной размерности
            min_len = min(len(vector_a), len(vector_b))
            vector_a = vector_a[:min_len]
            vector_b = vector_b[:min_len]
        
        if method == "cosine":
            return self._cosine_similarity(vector_a, vector_b)
        elif method == "euclidean":
            return self._euclidean_similarity(vector_a, vector_b)
        elif method == "jaccard":
            return self._jaccard_similarity(vector_a, vector_b)
        else:
            # Combined
            return self._combined_similarity(vector_a, vector_b)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Косинусная схожесть."""
        dot_product = sum(x * y for x, y in zip(a, b))
        
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _euclidean_similarity(self, a: List[float], b: List[float]) -> float:
        """Евклидово расстояние (инвертированное в схожесть)."""
        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        
        # Нормализуем к [0, 1]
        max_distance = math.sqrt(len(a))
        similarity = 1.0 - min(1.0, distance / max(max_distance, 1e-10))
        
        return max(0.0, similarity)
    
    def _jaccard_similarity(self, a: List[float], b: List[float]) -> float:
        """Схожесть Жаккара (по множествам ненулевых элементов)."""
        set_a = {i for i, x in enumerate(a) if abs(x) > 1e-10}
        set_b = {i for i, x in enumerate(b) if abs(x) > 1e-10}
        
        if not set_a and not set_b:
            return 1.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / max(1, union)
    
    def _combined_similarity(self, a: List[float], b: List[float]) -> float:
        """Комбинированная схожесть."""
        cosine = self._cosine_similarity(a, b)
        euclidean = self._euclidean_similarity(a, b)
        jaccard = self._jaccard_similarity(a, b)
        
        return (
            cosine * self.weights["cosine"] +
            euclidean * self.weights["euclidean"] +
            jaccard * self.weights["jaccard"]
        )
    
    def find_most_similar(
        self,
        query_vector: List[float],
        candidates: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Находит наиболее похожие векторы.
        
        Args:
            query_vector: Вектор запроса
            candidates: Список векторов-кандидатов
            top_k: Количество лучших результатов
            
        Returns:
            List[tuple]: [(index, similarity), ...] отсортированные по убыванию
        """
        similarities = [
            (i, self.compute(query_vector, candidate))
            for i, candidate in enumerate(candidates)
        ]
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

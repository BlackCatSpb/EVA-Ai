"""
SemanticEmbedder - создание векторных представлений запросов.
Использует простой но эффективный метод для семантического поиска.
"""
import hashlib
import logging
from typing import List, Optional
import math

logger = logging.getLogger("eva.memory.fractal_cache.embedder")


class SemanticEmbedder:
    """
    Создаёт векторные представления текста для семантического сравнения.
    Использует TF-IDF-like метод с локальной обработкой.
    """
    
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions
        
        # Словарь для частотного анализа
        self._word_freq = {}
        self._doc_count = 0
        
        logger.info(f"SemanticEmbedder инициализирован (dimensions={dimensions})")
    
    def encode(self, text: str) -> List[float]:
        """
        Создаёт векторное представление текста.
        
        Args:
            text: Входной текст
            
        Returns:
            List[float]: Вектор эмбеддинга
        """
        if not text:
            return [0.0] * self.dimensions
        
        # Токенизация
        tokens = self._tokenize(text)
        
        # Подсчёт частот
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        
        # Обновляем глобальные частоты
        for token in set(tokens):
            self._word_freq[token] = self._word_freq.get(token, 0) + 1
        self._doc_count += 1
        
        # Создаём вектор
        vector = [0.0] * self.dimensions
        
        for token, count in freq.items():
            # IDF вес
            idf = math.log(max(1, self._doc_count) / max(1, self._word_freq.get(token, 1)))
            
            # Позиция в векторе (хэш-маппинг)
            idx = self._hash_to_index(token)
            
            # TF-IDF вес
            tfidf = (count / max(1, len(tokens))) * idf
            vector[idx] += tfidf
        
        # Нормализация
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def _tokenize(self, text: str) -> List[str]:
        """Токенизация текста."""
        import re
        
        # Нижний регистр
        text = text.lower().strip()
        
        # Удаляем знаки препинания, оставляем слова
        tokens = re.findall(r'\w+', text)
        
        # Фильтруем короткие токены
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def _hash_to_index(self, token: str) -> int:
        """Хэширует токен в индекс вектора."""
        h = hashlib.md5(token.encode()).hexdigest()
        return int(h, 16) % self.dimensions
    
    def batch_encode(self, texts: List[str]) -> List[List[float]]:
        """
        Создаёт эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов
            
        Returns:
            List[List[float]]: Список векторов
        """
        return [self.encode(text) for text in texts]

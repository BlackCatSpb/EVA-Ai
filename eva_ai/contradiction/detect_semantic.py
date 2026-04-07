"""Semantic contradiction detection using embeddings."""
import logging
import re
import numpy as np
from typing import Dict, List, Any, Tuple

from eva_ai.nlp_fallbacks import compute_semantic_similarity, tokenize

logger = logging.getLogger("eva_ai.contradiction.detection.semantic")


class SemanticDetectionMixin:
    """Mixin providing semantic contradiction detection capabilities."""
    
    def _calculate_text_divergence(self, text1: str, text2: str) -> float:
        """
        Вычисляет семантическое расхождение между двумя текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            float: Уровень расхождения (0.0-1.0)
        """
        try:
            sim = float(compute_semantic_similarity([text1, text2], self.nlp_model))
            divergence = 1.0 - sim
            return max(0.0, min(1.0, divergence))
        except Exception as e:
            logger.error(f"Ошибка вычисления семантического расхождения: {e}")
            return 0.5
    
    def _calculate_lexical_divergence(self, text1: str, text2: str) -> float:
        """
        Вычисляет лексическое расхождение между двумя текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            float: Уровень расхождения (0.0-1.0)
        """
        def preprocess(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            words = [w for w in tokenize(text) if w.isalnum() and w not in self.stop_words]
            return set(words)
        
        words1 = preprocess(text1)
        words2 = preprocess(text2)
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 1.0
        
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity

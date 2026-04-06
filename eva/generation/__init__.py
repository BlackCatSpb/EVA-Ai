"""
Модуль генерации текста для ЕВА.

Содержит классы и утилиты для генерации текста с использованием языковых моделей,
включая управление кешированием, токенизацию и генерацию.
"""

from .generation_coordinator import UnifiedGenerationCoordinator as GenerationCoordinator
from ..memory.hybrid_token_cache import HybridTokenCache
from ..mlearning.parallel_tokenization import ParallelTokenizer

__all__ = [
    'GenerationCoordinator',
    'HybridTokenCache',
    'ParallelTokenizer'
]

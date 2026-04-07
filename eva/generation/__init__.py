"""
Модуль генерации текста для ЕВА.

Содержит классы и утилиты для генерации текста с использованием языковых моделей,
включая управление кешированием, токенизацию и генерацию.
"""

def __getattr__(name):
    if name == "GenerationCoordinator":
        from .generation_coordinator import UnifiedGenerationCoordinator
        return UnifiedGenerationCoordinator
    if name == "HybridTokenCache":
        try:
            from eva.memory.hybrid_token_cache import HybridTokenCache
        except ImportError:
            from eva.memory import HybridTokenCache
        return HybridTokenCache
    if name == "ParallelTokenizer":
        try:
            from eva.mlearning.parallel_tokenization import ParallelTokenizer
        except ImportError:
            ParallelTokenizer = None
        return ParallelTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return ["GenerationCoordinator", "HybridTokenCache", "ParallelTokenizer"]


__all__ = [
    'GenerationCoordinator',
    'HybridTokenCache',
    'ParallelTokenizer'
]

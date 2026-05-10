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
            from eva_ai.memory.hybrid_token_cache import HybridTokenCache
        except ImportError:
            from eva_ai.memory import HybridTokenCache
        return HybridTokenCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return ["GenerationCoordinator", "HybridTokenCache"]


__all__ = [
    'GenerationCoordinator',
    'HybridTokenCache',
]

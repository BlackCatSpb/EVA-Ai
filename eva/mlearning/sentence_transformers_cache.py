"""
Singleton кэш для sentence-transformers моделей
Избегает повторной загрузки модели (5+ секунд каждый раз)
"""
import logging
from typing import Optional

logger = logging.getLogger("eva.sentence_transformers_cache")

_SENTENCE_TRANSFORMER_CACHE: Optional[object] = None
_CACHE_MODEL_NAME: Optional[str] = None


def _detect_device() -> str:
    """Автоматически определяет устройство для загрузки."""
    try:
        import torch
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            total_mb = total_mem / (1024 ** 2)
            if total_mb >= 1500:
                logger.info(f"CUDA detected ({total_mb:.0f}MB), using GPU for embeddings")
                return "cuda"
            else:
                logger.info(f"CUDA detected but low memory ({total_mb:.0f}MB), using CPU")
                return "cpu"
    except Exception:
        pass
    return "cpu"


def get_sentence_transformer(model_name: str = "intfloat/multilingual-e5-small", device: str = "auto") -> Optional[object]:
    """
    Возвращает кэшированную модель sentence-transformers.
    Если модель уже загружена - возвращает кэш, иначе загружает и кэширует.
    
    Args:
        model_name: Имя модели (по умолчанию intfloat/multilingual-e5-small)
        device: Устройство ('cpu', 'cuda', 'auto')
    
    Returns:
        SentenceTransformer instance или None если загрузка не удалась
    """
    global _SENTENCE_TRANSFORMER_CACHE, _CACHE_MODEL_NAME
    
    if device == "auto":
        device = _detect_device()
    
    if _SENTENCE_TRANSFORMER_CACHE is not None and _CACHE_MODEL_NAME == model_name:
        logger.debug(f"Используем кэшированную модель sentence-transformers: {model_name}")
        return _SENTENCE_TRANSFORMER_CACHE
    
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Загрузка sentence-transformers модели: {model_name} (устройство: {device})")
        _SENTENCE_TRANSFORMER_CACHE = SentenceTransformer(model_name, device=device)
        _CACHE_MODEL_NAME = model_name
        logger.info(f"Модель sentence-transformers загружена и кэширована: {model_name} на {device}")
        return _SENTENCE_TRANSFORMER_CACHE
        
    except Exception as e:
        logger.warning(f"Не удалось загрузить sentence-transformers модель {model_name}: {e}")
        return None


def clear_sentence_transformer_cache():
    """Очищает кэш sentence-transformers."""
    global _SENTENCE_TRANSFORMER_CACHE, _CACHE_MODEL_NAME
    _SENTENCE_TRANSFORMER_CACHE = None
    _CACHE_MODEL_NAME = None
    logger.info("Кэш sentence-transformers очищен")


def is_sentence_transformer_loaded() -> bool:
    """Проверяет, загружена ли модель в кэш."""
    return _SENTENCE_TRANSFORMER_CACHE is not None
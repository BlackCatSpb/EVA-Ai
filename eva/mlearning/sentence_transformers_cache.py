"""
Singleton кэш для sentence-transformers моделей
Избегает повторной загрузки модели (5+ секунд каждый раз)
"""
import logging
from typing import Optional

logger = logging.getLogger("eva.sentence_transformers_cache")

_SENTENCE_TRANSFORMER_CACHE: Optional[object] = None
_CACHE_MODEL_NAME: Optional[str] = None


def get_sentence_transformer(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu") -> Optional[object]:
    """
    Возвращает кэшированную модель sentence-transformers.
    Если модель уже загружена - возвращает кэш, иначе загружает и кэширует.
    
    Args:
        model_name: Имя модели (по умолчанию paraphrase-multilingual-MiniLM-L12-v2)
        device: Устройство для загрузки ('cpu', 'cuda', 'cuda:0' и т.д.)
    
    Returns:
        SentenceTransformer instance или None если загрузка не удалась
    """
    global _SENTENCE_TRANSFORMER_CACHE, _CACHE_MODEL_NAME
    
    # Проверяем, используем ли ту же модель
    if _SENTENCE_TRANSFORMER_CACHE is not None and _CACHE_MODEL_NAME == model_name:
        logger.debug(f"Используем кэшированную модель sentence-transformers: {model_name}")
        return _SENTENCE_TRANSFORMER_CACHE
    
    # Загружаем новую модель
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Загрузка sentence-transformers модели: {model_name} (устройство: {device})")
        _SENTENCE_TRANSFORMER_CACHE = SentenceTransformer(model_name, device=device)
        _CACHE_MODEL_NAME = model_name
        logger.info(f"Модель sentence-transformers загружена и кэширована: {model_name}")
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
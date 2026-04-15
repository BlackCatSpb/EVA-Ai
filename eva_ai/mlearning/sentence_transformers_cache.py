"""
Singleton кэш для sentence-transformers моделей + кеш эмбеддингов
Избегает повторной загрузки модели (5+ секунд каждый раз)
Избегает повторного вычисления эмбеддингов (через EmbeddingCache)
"""
import os
import logging
from typing import Optional, List

logger = logging.getLogger("eva_ai.sentence_transformers_cache")

# Устанавливаем HF_HOME на локальный кеш
_DEFAULT_HF_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core', 'hf_cache')
if not os.environ.get('HF_HOME'):
    os.environ['HF_HOME'] = _DEFAULT_HF_CACHE
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    logger.info(f"HF_HOME установлен: {_DEFAULT_HF_CACHE}")

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


def _encode_with_cache(text: str, model) -> Optional[List[float]]:
    """Вычисляет эмбеддинг с кешированием."""
    try:
        from eva_ai.memory.embedding_cache import get_embedding_cache
        cache = get_embedding_cache()
        
        def compute_fn(t):
            emb = model.encode([t])[0]
            return emb.tolist() if hasattr(emb, 'tolist') else list(emb)
        
        return cache.get_or_compute(text, compute_fn)
    except Exception as e:
        logger.debug(f"EmbeddingCache недоступен: {e}")
        # Fallback без кеширования
        emb = model.encode([text])[0]
        return emb.tolist() if hasattr(emb, 'tolist') else list(emb)


def get_sentence_transformer(model_name: str = "eva_ai/core/hf_cache/multilingual-e5-base", device: str = "auto") -> Optional[object]:
    """
    Возвращает кэшированную модель sentence-transformers.
    Если модель уже загружена - возвращает кэш, иначе загружает и кэширует.
    
    Args:
        model_name: Путь к локальной модели или имя HF модели
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


def encode_text(text: str, model_name: str = "eva_ai/core/hf_cache/multilingual-e5-base", device: str = "auto") -> Optional[List[float]]:
    """
    Вычисляет эмбеддинг текста с автоматическим кешированием.

    Args:
        text: Текст для эмбеддинга
        model_name: Путь к локальной модели
        device: Устройство
    
    Returns:
        Embedding vector или None
    """
    model = get_sentence_transformer(model_name, device)
    if model is None:
        return None
    
    return _encode_with_cache(text, model)


def encode_batch(texts: List[str], model_name: str = "eva_ai/core/hf_cache/multilingual-e5-base", device: str = "auto") -> Optional[List[List[float]]]:
    """
    Вычисляет эмбеддинги батча текстов с кешированием.

    Returns:
        List of embedding vectors
    """
    model = get_sentence_transformer(model_name, device)
    if model is None:
        return None
    
    results = []
    for text in texts:
        emb = _encode_with_cache(text, model)
        if emb:
            results.append(emb)
        else:
            results.append(None)
    return results


def clear_sentence_transformer_cache():
    """Очищает кэш sentence-transformers."""
    global _SENTENCE_TRANSFORMER_CACHE, _CACHE_MODEL_NAME
    _SENTENCE_TRANSFORMER_CACHE = None
    _CACHE_MODEL_NAME = None
    logger.info("Кэш sentence-transformers очищен")


def is_sentence_transformer_loaded() -> bool:
    """Проверяет, загружена ли модель в кэш."""
    return _SENTENCE_TRANSFORMER_CACHE is not None
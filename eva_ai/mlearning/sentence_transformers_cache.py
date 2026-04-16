"""
Singleton кэш для sentence-transformers моделей + кеш эмбеддингов
Избегает повторной загрузки модели (5+ секунд каждый раз)
Избегает повторного вычисления эмбеддингов (через EmbeddingCache)
"""
import os
import logging
from typing import Optional, List

logger = logging.getLogger("eva_ai.sentence_transformers_cache")

# Путь к локальной модели
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'core', 'hf_cache')
_LOCAL_MODEL_PATH = os.path.join(_CACHE_DIR, 'models--intfloat--multilingual-e5-base', 'snapshots')

# Устанавливаем HF_HOME
os.environ['HF_HOME'] = _CACHE_DIR
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
logger.info(f"HF_HOME: {_CACHE_DIR}")
logger.info(f"LOCAL_MODEL_PATH: {_LOCAL_MODEL_PATH}")

# Найти конкретную папку snapshots
def _find_snapshot_path():
    if os.path.exists(_LOCAL_MODEL_PATH):
        for item in os.listdir(_LOCAL_MODEL_PATH):
            snapshot_path = os.path.join(_LOCAL_MODEL_PATH, item)
            if os.path.isdir(snapshot_path):
                return snapshot_path
    return None

_SNAPSHOT_PATH = _find_snapshot_path()
if _SNAPSHOT_PATH:
    logger.info(f"Found model snapshot: {_SNAPSHOT_PATH}")
else:
    logger.warning("Model snapshot not found in local cache!")

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


def get_sentence_transformer(model_name: str = None, device: str = "auto") -> Optional[object]:
    """
    Возвращает кэшированную модель sentence-transformers из ЛОКАЛЬНОГО кеша.
    Если модель уже загружена - возвращает кэш, иначе загружает и кэширует.
    
    Args:
        model_name: Не используется (для совместимости), модель всегда загружается из локального кеша
        device: Устройство ('cpu', 'cuda', 'auto')
    
    Returns:
        SentenceTransformer instance или None если загрузка не удалась
    """
    global _SENTENCE_TRANSFORMER_CACHE, _CACHE_MODEL_NAME
    
    if device == "auto":
        device = _detect_device()
    
    # Всегда используем локальный путь
    local_path = _SNAPSHOT_PATH
    
    if _SENTENCE_TRANSFORMER_CACHE is not None and _CACHE_MODEL_NAME == local_path:
        logger.debug(f"Используем кэшированную модель: {local_path}")
        return _SENTENCE_TRANSFORMER_CACHE
    
    try:
        from sentence_transformers import SentenceTransformer
        
        if local_path and os.path.exists(local_path):
            logger.info(f"Загрузка модели из ЛОКАЛЬНОГО кеша: {local_path} (устройство: {device})")
            _SENTENCE_TRANSFORMER_CACHE = SentenceTransformer(local_path, device=device)
        else:
            logger.warning(f"Локальная модель не найдена: {local_path}, пробуем HF...")
            _SENTENCE_TRANSFORMER_CACHE = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
        
        _CACHE_MODEL_NAME = local_path
        logger.info(f"Модель загружена на {device}")
        return _SENTENCE_TRANSFORMER_CACHE
        
    except Exception as e:
        logger.error(f"Не удалось загрузить sentence-transformers модель: {e}")
        return None


def encode_text(text: str, device: str = "auto") -> Optional[List[float]]:
    """
    Вычисляет эмбеддинг текста с автоматическим кешированием.

    Args:
        text: Текст для эмбеддинга
        device: Устройство
    
    Returns:
        Embedding vector или None
    """
    model = get_sentence_transformer(device=device)
    if model is None:
        return None
    
    return _encode_with_cache(text, model)


def encode_batch(texts: List[str], device: str = "auto") -> Optional[List[List[float]]]:
    """
    Вычисляет эмбеддинги батча текстов с кешированием.

    Returns:
        List of embedding vectors
    """
    model = get_sentence_transformer(device=device)
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
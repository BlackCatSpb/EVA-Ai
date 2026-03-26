"""
Конфигурация моделей для CogniFlex
"""

# Доступные модели
MODEL_CONFIGS = {
    "qwen3.5-0.8b": {
        "name": "qwen3.5-0.8b",
        "display_name": "Qwen 3.5 0.8B",
        "description": "Локальная модель Qwen 3.5 (0.8B параметров)",
        "requires_download": False,
        "size_mb": 1600,
        "supports_gpu": True,
        "supports_russian": True,
        "quality": 5
    },
    "qwen3.5-2b": {
        "name": "qwen3.5-2b",
        "display_name": "Qwen 3.5 2B",
        "description": "Локальная модель Qwen 3.5 (2B параметров)",
        "requires_download": True,
        "size_mb": 4200,
        "supports_gpu": True,
        "supports_russian": True,
        "quality": 5
    },
    "qwen3-1.8b": {
        "name": "qwen3-1.8b",
        "display_name": "Qwen 3 1.8B",
        "description": "Локальная модель Qwen 3 (1.8B параметров)",
        "requires_download": True,
        "size_mb": 3600,
        "supports_gpu": True,
        "supports_russian": True,
        "quality": 5
    }
}

# Рекомендации по выбору модели
MODEL_RECOMMENDATIONS = {
    "fast": "qwen3.5-0.8b",
    "balanced": "qwen3.5-0.8b", 
    "quality": "qwen3.5-2b",
    "russian": "qwen3.5-0.8b",
    "gpu": "qwen3.5-0.8b"
}

# Настройки по умолчанию
DEFAULT_MODEL = "qwen3.5-0.8b"
DEFAULT_SETTINGS = {
    "max_memory_gb": 1.5,
    "enable_gpu_tokenization": True,
    "cache_tokens": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,  # Was 40
    "no_repeat_ngram_size": 3,
    "max_new_tokens": 2048
}

def get_model_config(model_name: str) -> dict:
    """Получает конфигурацию модели."""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS[DEFAULT_MODEL])

def list_available_models() -> list:
    """Возвращает список доступных моделей."""
    return list(MODEL_CONFIGS.keys())

def get_recommended_model(preference: str) -> str:
    """Возвращает рекомендованную модель по предпочтению."""
    return MODEL_RECOMMENDATIONS.get(preference, DEFAULT_MODEL)

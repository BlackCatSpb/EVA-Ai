"""
Конфигурация моделей для CogniFlex
"""

# Доступные модели
MODEL_CONFIGS = {
    "gpt2": {
        "name": "gpt2",
        "display_name": "GPT-2",
        "description": "Локальная модель GPT-2 (124M параметров)",
        "requires_download": False,
        "size_mb": 500,
        "supports_gpu": True,
        "supports_russian": False,
        "quality": 3
    },
    "gpt2-medium": {
        "name": "gpt2-medium", 
        "display_name": "GPT-2 Medium",
        "description": "Локальная модель GPT-2 Medium (345M параметров)",
        "requires_download": True,
        "size_mb": 1500,
        "supports_gpu": True,
        "supports_russian": False,
        "quality": 4
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "display_name": "GPT-2 Large", 
        "description": "Локальная модель GPT-2 Large (774M параметров)",
        "requires_download": True,
        "size_mb": 3000,
        "supports_gpu": True,
        "supports_russian": False,
        "quality": 5
    },
    "rugpt3large": {
        "name": "sberbank-ai/rugpt3large_based_on_gpt2",
        "display_name": "ruGPT-3 Small",
        "description": "Русская модель от Сбера (125M параметров)",
        "requires_download": True,
        "size_mb": 600,
        "supports_gpu": True,
        "supports_russian": True,
        "quality": 6
    },
    "rugpt3large": {
        "name": "sberbank-ai/rugpt3large_based_on_gpt2",
        "display_name": "ruGPT-3 Medium",
        "description": "Русская модель от Сбера (355M параметров)",
        "requires_download": True,
        "size_mb": 1500,
        "supports_gpu": True,
        "supports_russian": True,
        "quality": 7
    }
}

# Рекомендации по выбору модели
MODEL_RECOMMENDATIONS = {
    "fast": "gpt2",
    "balanced": "gpt2-medium", 
    "quality": "rugpt3large",
    "russian": "rugpt3large",
    "gpu": "rugpt3large"
}

# Настройки по умолчанию
DEFAULT_MODEL = "rugpt3large"
DEFAULT_SETTINGS = {
    "max_memory_gb": 1.5,
    "enable_gpu_tokenization": True,
    "cache_tokens": True,
    "temperature": 0.4,
    "top_p": 0.75,
    "top_k": 40,
    "no_repeat_ngram_size": 3,
    "max_tokens": 200
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

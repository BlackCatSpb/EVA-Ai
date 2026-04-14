"""
Pie Architecture Model Paths Configuration

Конфигурация путей к моделям для Pie Architecture.
Модели остаются в eva_pie_architecture/models/

Актуальные модели:
- ruadapt_qwen3_4b: Основная модель для LOGIC и CONTEXT
- qwen_coder_1_5b: Модель для генерации кода
"""

from pathlib import Path
from typing import Dict, Optional

# Базовая директория моделей
PIE_MODELS_BASE = Path(r"C:\Users\black\OneDrive\Desktop\CogniFlex\eva_pie_architecture\models")

# Пути к конкретным моделям
PIE_MODEL_PATHS = {
    # Основная модель: ruadapt_qwen3_4b - CONDENSED (Logic)
    "ruadapt_qwen3_4b": {
        "condensed": PIE_MODELS_BASE / "gguf_models" / "ruadapt_qwen3_4b_q4_k_m.gguf",
    },
    # Extended модель: ruadapt_qwen3_4b - EXTENDED (Context) - ВТОРАЯ ФИЗИЧЕСКАЯ МОДЕЛЬ
    "ruadapt_qwen3_4b_extended": {
        "extended": PIE_MODELS_BASE / "gguf_models" / "ruadapt_qwen3_4b_extended" / "ruadapt_qwen3_4b_q4_k_m.gguf",
    },
    # Модель для кода: qwen_coder_1_5b
    "qwen_coder_1_5b": {
        "condensed": PIE_MODELS_BASE / "gguf_models" / "qwen2.5-coder-1.5b-instruct" / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    },
    # Embeddings модель
    "embeddings": {
        "semantic": PIE_MODELS_BASE / "embeddings" / "sentence-transformers" / "all-MiniLM-L6-v2",
    }
}


def get_pie_model_path(model_name: str, variant: str = "condensed") -> Optional[Path]:
    """
    Получить путь к модели Pie Architecture.
    
    Args:
        model_name: Имя модели ('ruadapt_qwen3_4b', 'qwen_coder_1_5b', etc.)
        variant: Вариант ('condensed', 'extended')
        
    Returns:
        Path к модели или None если не найдена
    """
    model_config = PIE_MODEL_PATHS.get(model_name)
    if not model_config:
        return None
    
    path = model_config.get(variant)
    if path and path.exists():
        return path
    
    return None


def get_available_pie_models() -> Dict[str, list]:
    """
    Получить список доступных моделей.
    
    Returns:
        Dict с именами моделей и их вариантами
    """
    available = {}
    
    for model_name, variants in PIE_MODEL_PATHS.items():
        available_variants = []
        for variant, path in variants.items():
            if isinstance(path, Path) and path.exists():
                available_variants.append(variant)
        
        if available_variants:
            available[model_name] = available_variants
    
    return available


def verify_pie_models() -> Dict[str, bool]:
    """
    Проверить доступность всех моделей.
    
    Returns:
        Dict с именами моделей и статусом доступности
    """
    status = {}
    
    for model_name, variants in PIE_MODEL_PATHS.items():
        for variant, path in variants.items():
            if isinstance(path, Path):
                key = f"{model_name}/{variant}"
                status[key] = path.exists()
    
    return status


# Для backward compatibility - маппинг на старые пути
def map_to_legacy_paths(pie_path: Path) -> Path:
    """
    Маппинг пути Pie Architecture на legacy пути eva_ai.
    
    Если нужно будет перенести модели позже.
    """
    # Пока просто возвращаем как есть
    return pie_path


if __name__ == "__main__":
    # Тестирование путей
    print("Pie Architecture Model Paths")
    print("=" * 60)
    
    print("\nAll configured paths:")
    for model_name, variants in PIE_MODEL_PATHS.items():
        print(f"\n{model_name}:")
        for variant, path in variants.items():
            if isinstance(path, Path):
                exists = "OK" if path.exists() else "MISSING"
                print(f"  [{exists}] {variant}: {path}")
            else:
                print(f"  [CFG] {variant}: {path}")
    
    print("\n" + "=" * 60)
    print("Available models:")
    available = get_available_pie_models()
    for model_name, variants in available.items():
        print(f"  {model_name}: {', '.join(variants)}")

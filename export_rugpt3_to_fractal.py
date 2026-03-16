#!/usr/bin/env python3
"""
Экспорт ruGPT-3 Medium в фрактальное хранилище
"""
import os
import json
import shutil
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_rugpt3_medium():
    """Экспортирует ruGPT-3 Medium в фрактальное хранилище"""
    
    # Пути
    source_path = "./models/rugpt3medium_fresh"
    target_path = "./fractal_storage_clean/rugpt3medium"
    
    # Проверяем исходную директорию
    if not os.path.exists(source_path):
        logger.error(f"Исходная директория не найдена: {source_path}")
        return False
    
    # Создаем целевую директорию
    os.makedirs(target_path, exist_ok=True)
    
    # Копируем модель
    model_source = os.path.join(source_path, "model")
    model_target = os.path.join(target_path, "model")
    
    if os.path.exists(model_source):
        if os.path.exists(model_target):
            shutil.rmtree(model_target)
        shutil.copytree(model_source, model_target)
        logger.info(f"Модель скопирована в {model_target}")
    
    # Копируем токенизатор
    tokenizer_source = os.path.join(source_path, "tokenizer")
    tokenizer_target = os.path.join(target_path, "tokenizer")
    
    if os.path.exists(tokenizer_source):
        if os.path.exists(tokenizer_target):
            shutil.rmtree(tokenizer_target)
        shutil.copytree(tokenizer_source, tokenizer_target)
        logger.info(f"Токенизатор скопирован в {tokenizer_target}")
    
    # Создаем метаданные
    metadata = {
        "model_name": "rugpt3medium",
        "model_type": "gpt2",
        "description": "ruGPT-3 Medium (355M параметров)",
        "vocab_size": 50257,
        "max_length": 2048,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "intermediate_size": 4096,
        "activation": "gelu_new",
        "dropout": 0.1,
        "version": "1.0",
        "export_date": "2026-03-09",
        "source": "local_export",
        "fractal_compatible": True,
        "memory_optimized": True
    }
    
    # Сохраняем метаданные
    with open(os.path.join(target_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Создаем конфигурацию для фрактального хранилища
    fractal_config = {
        "model_name": "rugpt3medium",
        "model_path": "./fractal_storage_clean/rugpt3medium",
        "config_path": "./fractal_storage_clean/rugpt3medium",
        "model_type": "fractal",
        "device": "auto",
        "use_fractal_storage": True,
        "memory_optimized": True,
        "cache_tokens": True,
        "max_memory_gb": 1.5,
        "vram_limit_gb": 1.5,
        "ram_limit_gb": 1.0,
        "ssd_limit_gb": 50.0
    }
    
    # Обновляем конфигурацию системы
    system_config_path = "./cogniflex/config/fractal_model_config.json"
    with open(system_config_path, "w", encoding="utf-8") as f:
        json.dump(fractal_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Конфигурация обновлена в {system_config_path}")
    
    # Проверяем результат
    if os.path.exists(os.path.join(target_path, "model", "model.safetensors")):
        logger.info("✅ Модель успешно экспортирована в фрактальное хранилище")
        return True
    else:
        logger.error("❌ Модель не найдена в фрактальном хранилище")
        return False

if __name__ == "__main__":
    success = export_rugpt3_medium()
    if success:
        print("Экспорт завершен успешно")
    else:
        print("Ошибка экспорта")

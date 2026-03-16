#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Импорт локальной модели ruGPT3 во фрактальное хранилище.
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("import_model.log")
    ]
)
logger = logging.getLogger("cogniflex.import_local_model")

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Импорты из проекта
try:
    from cogniflex.mlearning.storage.fractal_store import FractalWeightStore
except ImportError as e:
    logger.error(f"Ошибка импорта FractalWeightStore: {e}")
    sys.exit(1)

def setup_directories() -> bool:
    """Создаем необходимые директории."""
    try:
        base_dirs = [
            'ml_cache/models/fractal_rugpt',
            'ml_cache/models/hf_cache',
            'ml_cache/models/checkpoints'
        ]
        
        for dir_path in base_dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Проверена/создана директория: {dir_path}")
            
        return True
    except Exception as e:
        logger.error(f"Ошибка при создании директорий: {e}")
        return False

def copy_model_files(src_dir: Path, dest_dir: Path) -> bool:
    """Копируем файлы модели в целевую директорию."""
    try:
        # Создаем целевую директорию, если её нет
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Копируем все файлы из исходной директории
        for item in src_dir.glob('*'):
            if item.is_file():
                shutil.copy2(item, dest_dir / item.name)
                logger.info(f"Скопирован файл: {item.name}")
            elif item.is_dir():
                shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
                logger.info(f"Скопирована директория: {item.name}")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при копировании файлов модели: {e}")
        return False

def import_to_fractal(model_dir: Path, model_id: str = "rugpt3_large") -> bool:
    """Импортируем модель во фрактальное хранилище."""
    try:
        # Проверяем наличие необходимых файлов
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        missing_files = [f for f in required_files if not (model_dir / f).exists()]
        
        if missing_files:
            logger.error(f"Отсутствуют обязательные файлы модели: {', '.join(missing_files)}")
            return False
        
        # Инициализируем фрактальное хранилище
        logger.info("Инициализация фрактального хранилища...")
        fractal_store = FractalWeightStore()
        
        # Создаем временную директорию для упаковки
        temp_dir = Path("temp_import")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        try:
            # Копируем файлы модели во временную директорию
            if not copy_model_files(model_dir, temp_dir):
                return False
            
            # Упаковываем модель во фрактальное хранилище
            output_dir = Path(f"ml_cache/models/fractal_rugpt/{model_id}")
            
            logger.info(f"Упаковка модели {model_id} во фрактальное хранилище...")
            fractal_store.pack_model_weights(
                model_path=str(temp_dir),
                model_id=model_id,
                output_dir=str(output_dir)
            )
            
            logger.info(f"Модель успешно упакована в {output_dir}")
            return True
            
        finally:
            # Очищаем временную директорию
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
    except Exception as e:
        logger.error(f"Ошибка при импорте модели: {e}", exc_info=True)
        return False

def main():
    # Путь к локальной модели
    model_path = Path("C:/Users/black/OneDrive/Desktop/rugpt3_large")
    
    if not model_path.exists():
        logger.error(f"Директория с моделью не найдена: {model_path}")
        return 1
    
    logger.info(f"Найдена модель в директории: {model_path}")
    
    # Создаем необходимые директории
    if not setup_directories():
        return 1
    
    # Импортируем модель во фрактальное хранилище
    if not import_to_fractal(model_path):
        logger.error("Не удалось импортировать модель во фрактальное хранилище")
        return 1
    
    logger.info("Импорт модели успешно завершен!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

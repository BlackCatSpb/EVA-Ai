"""
Скрипт для импорта модели во фрактальное хранилище.
"""
import os
import sys
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories():
    """Создаем необходимые директории."""
    try:
        # Основные директории
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

def import_model():
    """Импортируем модель во фрактальное хранилище."""
    try:
        from cogniflex.mlearning.storage.fractal_store import FractalWeightStore
        from cogniflex.mlearning.model_manager import ModelManager
        
        # Инициализируем фрактальное хранилище
        logger.info("Инициализация фрактального хранилища...")
        fractal_store = FractalWeightStore()
        
        # Путь к модели (замените на актуальный путь к вашей модели)
        model_path = "sberbank-ai/rugpt3small_based_on_gpt2"
        
        # Импортируем модель
        logger.info(f"Импорт модели {model_path}...")
        
        # Используем ModelManager для загрузки и сохранения модели
        model_manager = ModelManager()
        
        # Загружаем модель (это может занять некоторое время)
        model, tokenizer = model_manager.load_model(
            model_name_or_path=model_path,
            use_fast_tokenizer=True,
            local_files_only=False,  # Разрешаем загрузку из интернета
            cache_dir="ml_cache/models/hf_cache"
        )
        
        # Сохраняем модель во фрактальное хранилище
        output_dir = "ml_cache/models/fractal_rugpt/rugpt3small"
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем модель и токенизатор
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Модель успешно сохранена в {output_dir}")
        
        # Упаковываем модель во фрактальное хранилище
        logger.info("Упаковка модели во фрактальное хранилище...")
        fractal_store.pack_model_weights(
            model_path=output_dir,
            model_id="rugpt3small",
            output_dir=output_dir + "_fractal"
        )
        
        logger.info("Модель успешно упакована во фрактальное хранилище!")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при импорте модели: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Начало настройки модели...")
    
    # Создаем необходимые директории
    if not setup_directories():
        logger.error("Не удалось создать необходимые директории")
        sys.exit(1)
    
    # Импортируем модель
    if import_model():
        logger.info("Настройка модели успешно завершена!")
    else:
        logger.error("Не удалось завершить настройку модели")
        sys.exit(1)

"""
Скрипт для проверки доступных моделей во фрактальном хранилище.
"""
import logging
import os
from pathlib import Path
from cogniflex.mlearning.storage.fractal_store import FractalWeightStore

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_fractal_store():
    try:
        # Инициализация фрактального хранилища
        logger.info("Инициализация фрактального хранилища...")
        fractal_store = FractalWeightStore()
        
        # Проверяем атрибуты и методы объекта
        logger.info("Доступные атрибуты FractalWeightStore:")
        for attr in dir(fractal_store):
            if not attr.startswith('_'):  # Пропускаем приватные атрибуты
                logger.info(f"- {attr}")
        
        # Проверяем наличие стандартных методов
        if hasattr(fractal_store, 'get_model_path'):
            logger.info("\nПроверка метода get_model_path:")
            try:
                model_path = fractal_store.get_model_path('ruGPT3Small')
                logger.info(f"Путь к модели ruGPT3Small: {model_path}")
            except Exception as e:
                logger.warning(f"Ошибка при вызове get_model_path: {e}")
        
        # Проверяем наличие стандартных директорий
        logger.info("\nПроверка стандартных директорий:")
        base_dir = Path('ml_cache/models/fractal_rugpt')
        if base_dir.exists():
            logger.info(f"Найдена базовая директория: {base_dir}")
            models = [d for d in base_dir.iterdir() if d.is_dir()]
            if models:
                logger.info("\nНайдены модели во фрактальном хранилище:")
                for model_dir in models:
                    logger.info(f"- {model_dir.name}")
                    # Выводим содержимое директории модели
                    for item in model_dir.glob('*'):
                        if item.is_file():
                            logger.info(f"  - {item.name} (файл, {item.stat().st_size} байт)")
                        elif item.is_dir():
                            logger.info(f"  - {item.name}/ (директория)")
            else:
                logger.warning(f"В директории {base_dir} нет поддиректорий с моделями.")
        else:
            logger.warning(f"Базовая директория {base_dir} не найдена.")
            
        logger.info("\nПроверка завершена.")
        
    except Exception as e:
        logger.error(f"Ошибка при проверке фрактального хранилища: {e}", exc_info=True)

if __name__ == "__main__":
    check_fractal_store()

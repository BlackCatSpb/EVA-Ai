"""
Скрипт для настройки токенизатора для модели fractal_unified_text-generation.
Проверяет наличие и валидность файлов токенизатора и при необходимости генерирует их заново.
"""

import os
import sys
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_tokenizer_files(model_dir: str) -> bool:
    """Проверяет наличие и валидность файлов токенизатора."""
    required_files = [
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'merges.txt'
    ]
    
    # Проверяем наличие всех файлов
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(model_dir, f))]
    if missing_files:
        logger.warning(f"Отсутствуют файлы токенизатора: {', '.join(missing_files)}")
        return False
    
    # Проверяем валидность JSON-файлов
    try:
        for f in ['tokenizer_config.json', 'special_tokens_map.json', 'vocab.json']:
            with open(os.path.join(model_dir, f), 'r', encoding='utf-8') as fh:
                json.load(fh)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при чтении файла {f}: {e}")
        return False
    
    return True

def generate_tokenizer(model_dir: str) -> bool:
    """Генерирует файлы токенизатора."""
    try:
        logger.info(f"Загрузка токенизатора из {model_dir}")
        
        # Загружаем токенизатор из существующих файлов
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=False
        )
        
        # Сохраняем токенизатор обратно (это обновит все необходимые файлы)
        logger.info("Сохранение обновленного токенизатора...")
        tokenizer.save_pretrained(model_dir)
        
        # Проверяем, что файлы созданы корректно
        return check_tokenizer_files(model_dir)
        
    except Exception as e:
        logger.error(f"Ошибка при генерации токенизатора: {e}", exc_info=True)
        return False

def main():
    # Путь к директории с моделью
    project_root = Path(__file__).parent.parent
    model_dir = project_root / 'cogniflex' / 'mlearning' / 'cogniflex_models' / 'fractal_unified_text-generation'
    
    if not model_dir.exists():
        logger.error(f"Директория с моделью не найдена: {model_dir}")
        return 1
    
    logger.info(f"Проверка токенизатора в {model_dir}")
    
    # Проверяем токенизатор
    if check_tokenizer_files(str(model_dir)):
        logger.info("Токенизатор в порядке, дополнительная настройка не требуется")
        return 0
    
    # Пытаемся сгенерировать токенизатор
    logger.warning("Обнаружены проблемы с токенизатором. Пытаемся сгенерировать заново...")
    if generate_tokenizer(str(model_dir)):
        logger.info("Токенизатор успешно сгенерирован")
        return 0
    else:
        logger.error("Не удалось сгенерировать токенизатор")
        return 1

if __name__ == "__main__":
    sys.exit(main())

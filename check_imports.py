#!/usr/bin/env python3
"""
Проверка всех импортов в системе CogniFlex на предмет rugpt3medium
"""
import os
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file(filepath):
    """Проверяет файл на наличие rugpt3medium"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ищем упоминания rugpt3medium
        patterns = [
            r'rugpt3medium',
            r'fractal_storage_clean',
            r'sberbank-ai/rugpt3medium_based_on_gpt2'
        ]
        
        issues = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                issues.extend(matches)
        
        if issues:
            logger.warning(f"ФАЙЛ: {filepath}")
            for issue in issues:
                logger.warning(f"  НАЙДЕНО: {issue}")
            return True
        return False
    except Exception as e:
        logger.error(f"Ошибка чтения файла {filepath}: {e}")
        return False

def main():
    """Основная функция проверки"""
    logger.info("🔍 ПРОВЕРКА ИМПОРТОВ НА RUGPT3MEDIUM")
    logger.info("=" * 50)
    
    # Ищем все .py файлы
    python_files = list(Path('.').rglob('*.py'))
    
    total_files = 0
    problematic_files = 0
    
    for filepath in python_files:
        total_files += 1
        if check_file(filepath):
            problematic_files += 1
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 РЕЗУЛЬТАТЫ:")
    logger.info(f"   Всего файлов: {total_files}")
    logger.info(f"   Проблемных файлов: {problematic_files}")
    
    if problematic_files == 0:
        logger.info("✅ ВСЕ ФАЙЛЫ ЧИСТЫ!")
    else:
        logger.warning(f"⚠️ НАЙДЕНО {problematic_files} ФАЙЛОВ С RUGPT3MEDIUM")

if __name__ == "__main__":
    main()

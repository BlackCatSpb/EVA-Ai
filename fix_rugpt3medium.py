#!/usr/bin/env python3
"""
Исправление всех упоминаний rugpt3medium на rugpt3large в системе CogniFlex
"""
import os
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_file(filepath):
    """Исправляет файл, заменяя rugpt3medium на rugpt3large"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Заменяем упоминания rugpt3medium на rugpt3large
        replacements = [
            (r'rugpt3medium', 'rugpt3large'),
            (r'fractal_storage_clean', 'cogniflex_cache/ml_unit/fractal_storage'),
            (r'sberbank-ai/rugpt3medium_based_on_gpt2', 'sberbank-ai/rugpt3large_based_on_gpt2'),
            (r'RuGPT3Medium', 'RuGPT3Large'),
            (r'ruGPT3Medium', 'ruGPT3Large'),
        ]
        
        changes_made = False
        for pattern, replacement in replacements:
            new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            if new_content != content:
                content = new_content
                changes_made = True
                logger.info(f"  Заменено: {pattern} -> {replacement}")
        
        if changes_made:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✅ ИСПРАВЛЕНО: {filepath}")
            return True
        else:
            logger.info(f"ℹ️ Без изменений: {filepath}")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка исправления файла {filepath}: {e}")
        return False

def main():
    """Основная функция исправления"""
    logger.info("🔧 ИСПРАВЛЕНИЕ RUGPT3MEDIUM НА RUGPT3LARGE")
    logger.info("=" * 60)
    
    # Файлы с проблемами (из лога проверки)
    problematic_files = [
        "cogniflex/mlearning/fractal_model_manager.py",
        "cogniflex/mlearning/fractal_rugpt3_manager.py", 
        "cogniflex/mlearning/local_rugpt3_loader.py",
        "cogniflex/mlearning/ml_unit.py",
        "cogniflex/mlearning/model_config.py",
        "cogniflex/mlearning/rugpt3_model_manager.py",
        "cogniflex/nlp/text_processor.py",
        "cogniflex/mlearning/storage/fractal_store_backup.py",
        "cogniflex/mlearning/storage/optimized_fractal_model_manager.py",
    ]
    
    total_files = 0
    fixed_files = 0
    
    for filepath in problematic_files:
        if os.path.exists(filepath):
            total_files += 1
            logger.info(f"\n📁 Обработка файла: {filepath}")
            if fix_file(filepath):
                fixed_files += 1
        else:
            logger.warning(f"⚠️ Файл не найден: {filepath}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 РЕЗУЛЬТАТЫ:")
    logger.info(f"   Всего файлов: {total_files}")
    logger.info(f"   Исправлено файлов: {fixed_files}")
    
    if fixed_files > 0:
        logger.info("✅ ИСПРАВЛЕНИЯ ВНЕСЕНЫ!")
    else:
        logger.info("ℹ️ ФАЙЛЫ НЕ ТРЕБОВАЛИ ИСПРАВЛЕНИЙ")

if __name__ == "__main__":
    main()

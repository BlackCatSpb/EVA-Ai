import os
import shutil
from pathlib import Path

def copy_tokenizer_files():
    # Путь к исходной директории с файлами токенизатора
    source_dir = Path(r"C:\Users\black\OneDrive\Desktop\CogniFlex\tokenizer_cache")
    
    # Путь к целевой директории
    target_dir = Path(r"C:\Users\black\OneDrive\Desktop\CogniFlex\cogniflex\mlearning\tokenizers\fractal_unified_tokenizer")
    
    # Создаем целевую директорию, если она не существует
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Список файлов для копирования
    files_to_copy = [
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt'
    ]
    
    # Копируем файлы
    for file_name in files_to_copy:
        source_file = source_dir / file_name
        target_file = target_dir / file_name
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"Скопирован файл: {file_name}")
        else:
            print(f"Внимание: файл {file_name} не найден в исходной директории")
    
    print("\nКопирование файлов токенизатора завершено!")

if __name__ == "__main__":
    copy_tokenizer_files()

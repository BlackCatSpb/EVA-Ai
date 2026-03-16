#!/usr/bin/env python3
"""Скрипт для исправления конфликтов слияния во всем проекте CogniFlex"""

import os
import re
import sys

def find_files_with_conflicts(root_dir):
    """Находит все Python файлы с маркерами конфликтов"""
    conflict_files = []
    for root, dirs, files in os.walk(root_dir):
        # Исключаем __pycache__ и .git
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.venv', 'venv']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if '<<<<<<<' in content or '=======' in content or '>>>>>>>' in content:
                        conflict_files.append(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    return conflict_files

def fix_merge_conflicts(filepath):
    """Исправляет конфликты слияния в файле - выбирает новую версию (после =======)"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    
    result = []
    i = 0
    fixed = False
    
    while i < len(lines):
        line = lines[i]
        
        # Ищем начало конфликта <<<<<<<
        if line.startswith('<<<<<<<'):
            # Пропускаем строки до =======
            i += 1
            while i < len(lines) and not lines[i].startswith('======='):
                i += 1
            
            # Теперь берем строки после ======= до >>>>>>>
            if i < len(lines) and lines[i].startswith('======='):
                i += 1
                # Добавляем строки из новой версии
                while i < len(lines) and not lines[i].startswith('>>>>>>>'):
                    result.append(lines[i])
                    i += 1
                
                # Пропускаем >>>>>>>
                if i < len(lines):
                    i += 1
                fixed = True
        else:
            result.append(line)
            i += 1
    
    if fixed:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(result)
            return True
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return False
    
    return False

def main():
    root_dir = r'c:\Users\black\OneDrive\Desktop\CogniFlex\cogniflex'
    
    print("Поиск файлов с конфликтами слияния...")
    conflict_files = find_files_with_conflicts(root_dir)
    
    if not conflict_files:
        print("Файлы с конфликтами не найдены")
        return 0
    
    print(f"\nНайдено {len(conflict_files)} файлов с конфликтами:")
    for f in conflict_files:
        print(f"  - {f}")
    
    print("\nИсправление конфликтов...")
    fixed_count = 0
    for filepath in conflict_files:
        if fix_merge_conflicts(filepath):
            print(f"  Исправлен: {filepath}")
            fixed_count += 1
        else:
            print(f"  Не удалось исправить: {filepath}")
    
    print(f"\nИсправлено {fixed_count} из {len(conflict_files)} файлов")
    return 0

if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Скрипт для проверки всех импортов в проекте CogniFlex"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict

def find_all_python_files(root_dir):
    """Находит все Python файлы в проекте"""
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        # Исключаем служебные директории
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.venv', 'venv', 'text-generation']]
        for filename in filenames:
            if filename.endswith('.py'):
                files.append(os.path.join(root, filename))
    return files

def extract_imports(filepath):
    """Извлекает все импорты из файла"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        
        tree = ast.parse(source)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if module:
                        imports.append(f"{module}.{alias.name}")
                    else:
                        imports.append(alias.name)
        
        return imports
    except Exception as e:
        return [f"ERROR: {e}"]

def check_cogniflex_imports(files):
    """Проверяет импорты CogniFlex и находит потенциальные проблемы"""
    issues = []
    cogniflex_modules = set()
    
    # Сначала собираем все модули CogniFlex
    for filepath in files:
        rel_path = os.path.relpath(filepath, r'c:\Users\black\OneDrive\Desktop\CogniFlex')
        if rel_path.startswith('cogniflex'):
            # Преобразуем путь в имя модуля
            module_name = rel_path.replace(os.sep, '.').replace('.py', '')
            cogniflex_modules.add(module_name)
    
    # Теперь проверяем импорты
    for filepath in files:
        if not filepath.startswith(r'c:\Users\black\OneDrive\Desktop\CogniFlex\cogniflex'):
            continue
            
        imports = extract_imports(filepath)
        rel_path = os.path.relpath(filepath, r'c:\Users\black\OneDrive\Desktop\CogniFlex')
        
        for imp in imports:
            if imp.startswith('cogniflex.') or imp.startswith('..'):
                # Проверяем, существует ли модуль
                if imp.startswith('..'):
                    # Относительный импорт - проверяем сложнее
                    parts = rel_path.replace('.py', '').split(os.sep)
                    if imp.startswith('...'):
                        # Импорт из родителя родителя
                        base = '.'.join(parts[:-3])
                        rest = imp[3:]
                    else:
                        # Импорт из родителя
                        base = '.'.join(parts[:-2])
                        rest = imp[2:]
                    
                    if rest:
                        full_module = f"{base}.{rest}" if base else rest
                    else:
                        full_module = base
                else:
                    full_module = imp
                
                # Проверяем существование модуля
                module_path = full_module.replace('.', os.sep) + '.py'
                init_path = full_module.replace('.', os.sep) + os.sep + '__init__.py'
                
                full_module_path = os.path.join(r'c:\Users\black\OneDrive\Desktop\CogniFlex', module_path)
                full_init_path = os.path.join(r'c:\Users\black\OneDrive\Desktop\CogniFlex', init_path)
                
                if not os.path.exists(full_module_path) and not os.path.exists(full_init_path):
                    issues.append(f"{rel_path}: импорт {imp} -> модуль не найден: {full_module}")
    
    return issues

if __name__ == '__main__':
    print("Поиск всех Python файлов...")
    files = find_all_python_files(r'c:\Users\black\OneDrive\Desktop\CogniFlex')
    print(f"Найдено {len(files)} файлов")
    
    print("\nПроверка импортов CogniFlex...")
    issues = check_cogniflex_imports(files)
    
    if issues:
        print(f"\nНайдено {len(issues)} проблем с импортами:")
        for issue in issues[:50]:  # Показываем первые 50
            print(f"  - {issue}")
        if len(issues) > 50:
            print(f"  ... и еще {len(issues) - 50} проблем")
    else:
        print("\nПроблем с импортами не найдено!")

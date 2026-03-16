#!/usr/bin/env python3
"""
Скрипт для исправления ошибок в knowledge_graph.py
Исправляет проблемы с json.loads и tuple index out of range
"""

import re
import os
import json

def safe_json_loads(value):
    """Безопасная загрузка JSON с обработкой ошибок."""
    if not value:
        return {}
    try:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, (bytes, bytearray)):
            return json.loads(value.decode('utf-8'))
        else:
            # Если значение не является строкой, возвращаем пустой dict
            return {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}

def fix_json_parsing(file_path):
    """Исправляет проблемные строки с json.loads."""
    print(f"Исправляем файл: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Заменяем проблемные строки на безопасные версии
    replacements = [
        # Основная проблема: json.loads(row[9]) без проверок
        (r'meta=json\.loads\(row\[9\]\) if row\[9\] else \{\},',
         r'meta=safe_json_loads(row[9]) if len(row) > 9 and row[9] else {},'),

        # Также исправляем другие json.loads с проверками
        (r'spatial_info=json\.loads\(row\[10\]\) if row\[10\] else \{\},',
         r'spatial_info=safe_json_loads(row[10]) if len(row) > 10 and row[10] else {},'),

        (r'temporal_info=json\.loads\(row\[11\]\) if row\[11\] else \{\},',
         r'temporal_info=safe_json_loads(row[11]) if len(row) > 11 and row[11] else {},'),
    ]

    for old, new in replacements:
        content = re.sub(old, new, content)
        print(f"  Заменено: {old} -> {new}")

    # Добавляем функцию safe_json_loads в начало файла
    safe_function = '''
def safe_json_loads(value):
    """Безопасная загрузка JSON с обработкой ошибок."""
    if not value:
        return {}
    try:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, (bytes, bytearray)):
            return json.loads(value.decode('utf-8'))
        else:
            # Если значение не является строкой, возвращаем пустой dict
            return {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}
'''

    # Находим место для вставки функции (после импортов)
    lines = content.split('\n')
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_pos = i + 1
        elif line.strip() and not line.startswith('import ') and not line.startswith('from ') and not line.startswith('#'):
            break

    # Вставляем функцию
    lines.insert(insert_pos, safe_function)
    content = '\n'.join(lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ Исправлен файл: {file_path}")

# Находим и исправляем knowledge_graph.py
for root, dirs, files in os.walk('.'):
    for file in files:
        if file == 'knowledge_graph.py':
            filepath = os.path.join(root, file)
            fix_json_parsing(filepath)
            print(f"✅ Найден и исправлен: {filepath}")
            break
else:
    print("❌ knowledge_graph.py не найден")

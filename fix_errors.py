#!/usr/bin/env python3
"""
Быстрое исправление ошибок в knowledge_graph.py
"""

import os

def fix_knowledge_graph():
    file_path = 'cogniflex/knowledge/knowledge_graph.py'

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Добавляем функцию safe_json_loads если её нет
    if 'def safe_json_loads' not in content:
        safe_func = '''def safe_json_loads(value):
    """Безопасная загрузка JSON с обработкой ошибок."""
    if not value:
        return {}
    try:
        if isinstance(value, str):
            return json.loads(value)
        elif isinstance(value, (bytes, bytearray)):
            return json.loads(value.decode('utf-8'))
        else:
            return {}
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return {}

'''
        # Находим место после импортов
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
            elif line.strip() and not line.startswith('import ') and not line.startswith('from ') and not line.startswith('#'):
                break

        lines.insert(insert_pos, safe_func)
        content = '\n'.join(lines)
        print('✅ Добавлена функция safe_json_loads')

    # Заменяем проблемные строки
    replacements = [
        ('node.history = json.loads(row[12]) if row[12] else []',
         'node.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []'),

        ('node.contradictions = json.loads(row[13]) if row[13] else []',
         'node.contradictions = safe_json_loads(row[13]) if len(row) > 13 and row[13] else []'),

        ('node.keyword_index = json.loads(row[14]) if row[14] else []',
         'node.keyword_index = safe_json_loads(row[14]) if len(row) > 14 and row[14] else []'),

        ('node.concept_index = json.loads(row[15]) if row[15] else []',
         'node.concept_index = safe_json_loads(row[15]) if len(row) > 15 and row[15] else []'),

        ('edge.history = json.loads(row[12]) if row[12] else []',
         'edge.history = safe_json_loads(row[12]) if len(row) > 12 and row[12] else []'),

        ('"changes": json.loads(row[4]) if row[4] else {},',
         '"changes": safe_json_loads(row[4]) if len(row) > 4 and row[4] else {},'),
    ]

    fixed = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            fixed += 1
            print(f'✅ Исправлено: {old}')

    print(f'\\n📊 Всего исправлено: {fixed} строк')

    # Сохраняем файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print('\\n🎉 Файл knowledge_graph.py полностью исправлен!')

if __name__ == '__main__':
    os.chdir('C:\\Users\\black\\OneDrive\\Desktop\\CogniFlex')
    fix_knowledge_graph()

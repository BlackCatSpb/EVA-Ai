#!/usr/bin/env python3
"""
Проверка интеграции всех модулей с BaseComponent
"""

import os
import sys

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniflex.core.base_component import BaseComponent

def find_classes_in_file(file_path):
    """Находит классы в файле"""
    classes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('class ') and '(' in line:
                class_name = line.split('class ')[1].split('(')[0].strip()
                class_line = i + 1
                classes.append((class_name, class_line))
    except Exception as e:
        pass
    return classes

def check_base_component_integration():
    """Проверяет интеграцию с BaseComponent"""
    print('🔍 Проверка интеграции с BaseComponent:')
    print('=' * 60)
    
    # Ищем основные файлы компонентов
    component_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                if any(keyword in root.lower() for keyword in ['adaptation', 'analytics', 'contradiction', 'ethics', 'learning', 'websearch', 'text_processor', 'memory', 'knowledge']):
                    component_files.append(os.path.join(root, file))
    
    issues_found = []
    checked_files = 0
    
    for file_path in component_files[:30]:  # Ограничим для скорости
        classes = find_classes_in_file(file_path)
        if not classes:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_issues = []
            for class_name, line_num in classes:
                # Проверяем наследование от BaseComponent
                class_start = content.find(f'class {class_name}')
                if class_start == -1:
                    continue
                    
                # Находим конец объявления класса
                class_end = content.find(':', class_start)
                if class_end == -1:
                    continue
                    
                class_declaration = content[class_start:class_end + 1]
                
                rel_path = os.path.relpath(file_path, '.')
                
                if 'BaseComponent' in class_declaration:
                    print(f'  ✅ {class_name} ({rel_path}:{line_num}) - наследует BaseComponent')
                    
                    # Проверяем вызов super().__init__
                    class_body_start = class_end + 1
                    next_brace = content.find('def __init__', class_body_start)
                    if next_brace != -1:
                        init_start = next_brace
                        init_body = content.find('def ', init_start + 1)
                        if init_body == -1:
                            init_body = len(content)
                        
                        init_content = content[init_start:init_body]
                        if 'super().__init__' in init_content:
                            print(f'    ✅ Вызывает super().__init__')
                        else:
                            print(f'    ❌ НЕ вызывает super().__init__')
                            file_issues.append(f'{class_name} (нет super().__init__)')
                    else:
                        print(f'    ⚠️  Нет метода __init__')
                        file_issues.append(f'{class_name} (нет __init__)')
                else:
                    print(f'  ❌ {class_name} ({rel_path}:{line_num}) - НЕ наследует BaseComponent')
                    file_issues.append(f'{class_name} (нет BaseComponent)')
            
            if file_issues:
                issues_found.extend([f'{rel_path}: {issue}' for issue in file_issues])
            
            checked_files += 1
            
        except Exception as e:
            print(f'  ❌ Ошибка обработки {file_path}: {e}')
    
    print('\n' + '=' * 60)
    print(f'📊 Проверено файлов: {checked_files}')
    
    if issues_found:
        print(f'❌ Найдено проблем: {len(issues_found)}')
        print('\nПроблемные файлы:')
        for issue in issues_found:
            print(f'  - {issue}')
    else:
        print('✅ Все классы правильно интегрированы с BaseComponent')
    
    return len(issues_found) == 0

def main():
    """Основная функция"""
    success = check_base_component_integration()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

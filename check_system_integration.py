#!/usr/bin/env python3
"""
Комплексная проверка системы CogniFlex на предмет неинтегрированных компонентов
"""

import os
import sys
import time

# Добавляем путь к CogniFlex
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_system_integration():
    """Проверяет полную интеграцию системы"""
    
    print("🔍 Комплексная проверка интеграции системы CogniFlex...")
    
    # 1. Проверка структуры директорий
    print("\n1. 📁 Проверка структуры директорий...")
    base_dir = "cogniflex"
    expected_dirs = [
        "core", "memory", "knowledge", "learning", "analytics", "ethics",
        "gui", "nlp", "generation", "websearch", "contradiction", "adaptation",
        "mlearning", "monitoring", "security", "tools", "config", "storage"
    ]
    
    existing_dirs = []
    missing_dirs = []
    
    for dir_name in expected_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            existing_dirs.append(dir_name)
            # Проверяем наличие файлов
            files = [f for f in os.listdir(dir_path) if f.endswith('.py') and not f.startswith('__')]
            print(f"   ✅ {dir_name}: {len(files)} Python файлов")
        else:
            missing_dirs.append(dir_name)
            print(f"   ❌ {dir_name}: отсутствует")
    
    print(f"\n   📊 Существующих директорий: {len(existing_dirs)}")
    print(f"   ❌ Отсутствующих директорий: {len(missing_dirs)}")
    
    # 2. Проверка ComponentInitializer
    print("\n2. 🏗️ Проверка ComponentInitializer...")
    try:
        from cogniflex.core.component_initializer import ComponentInitializer
        from cogniflex.core.core_brain import CoreBrain
        
        brain = CoreBrain()
        initializer = ComponentInitializer(brain)
        
        print("   ✅ ComponentInitializer инициализирован")
        
        # Проверяем зарегистрированные фабрики
        factories = list(initializer.component_factories.keys())
        print(f"   🏭 Зарегистрировано фабрик: {len(factories)}")
        
        # Проверяем зависимости
        dependencies = initializer.component_dependencies
        print(f"   🔗 Настроено зависимостей: {len(dependencies)}")
        
        # Показываем все компоненты
        print("\n   📋 Список зарегистрированных компонентов:")
        for i, comp in enumerate(sorted(factories), 1):
            deps = dependencies.get(comp, [])
            deps_str = f" (зависит от: {', '.join(deps)})" if deps else " (без зависимостей)"
            print(f"   {i:2d}. {comp}{deps_str}")
        
    except Exception as e:
        print(f"   ❌ Ошибка ComponentInitializer: {e}")
        return False
    
    # 3. Проверка инициализации компонентов
    print("\n3. 🚀 Проверка инициализации компонентов...")
    try:
        # Инициализируем все компоненты
        success = initializer.initialize_components()
        
        if success:
            print("   ✅ Все компоненты успешно инициализированы")
            
            # Проверяем статистику
            stats = initializer.get_component_stats()
            print(f"   📊 Статистика:")
            print(f"      Всего компонентов: {stats['total_components']}")
            print(f"      Инициализировано: {stats['initialized_components']}")
            print(f"      С ошибками: {stats['failed_components']}")
            print(f"      Зарегистрировано экземпляров: {stats['registered_instances']}")
            
            # Проверяем компоненты в brain
            brain_components = list(brain.components.keys())
            print(f"   🧠 Компонентов в brain: {len(brain_components)}")
            
        else:
            print("   ❌ Ошибка инициализации компонентов")
            
    except Exception as e:
        print(f"   ❌ Ошибка инициализации: {e}")
        return False
    
    # 4. Проверка каждого компонента
    print("\n4. 🔍 Детальная проверка компонентов...")
    component_status = {}
    
    for comp_name in sorted(factories):
        try:
            component = brain.components.get(comp_name)
            if component:
                # Проверяем базовые методы
                methods = []
                if hasattr(component, 'start'):
                    methods.append('start')
                if hasattr(component, 'stop'):
                    methods.append('stop')
                if hasattr(component, 'initialize'):
                    methods.append('initialize')
                if hasattr(component, 'get_status'):
                    methods.append('get_status')
                if hasattr(component, 'is_ready'):
                    methods.append('is_ready')
                
                component_status[comp_name] = {
                    'status': '✅',
                    'type': type(component).__name__,
                    'methods': methods
                }
                print(f"   ✅ {comp_name}: {type(component).__name__} ({len(methods)} методов)")
            else:
                # Проверяем в component_instances
                component = initializer.component_instances.get(comp_name)
                if component:
                    component_status[comp_name] = {
                        'status': '⚠️',
                        'type': type(component).__name__,
                        'methods': [],
                        'note': 'в component_instances но не в brain.components'
                    }
                    print(f"   ⚠️ {comp_name}: {type(component).__name__} (в component_instances но не в brain.components)")
                else:
                    component_status[comp_name] = {
                        'status': '❌',
                        'type': 'None',
                        'methods': []
                    }
                    print(f"   ❌ {comp_name}: не найден")
                
        except Exception as e:
            component_status[comp_name] = {
                'status': '⚠️',
                'type': 'Error',
                'methods': [],
                'error': str(e)
            }
            print(f"   ⚠️ {comp_name}: ошибка - {e}")
    
    # 5. Проверка специфичных модулей
    print("\n5. 📦 Проверка специфичных модулей...")
    
    # Проверка GUI
    try:
        from cogniflex.gui.main_window import MainWindow
        print("   ✅ GUI MainWindow доступен")
    except Exception as e:
        print(f"   ❌ GUI MainWindow: {e}")
    
    # Проверка NLP
    try:
        from cogniflex.nlp.text_processor import UnifiedTextProcessor
        print("   ✅ NLP UnifiedTextProcessor доступен")
    except Exception as e:
        print(f"   ❌ NLP UnifiedTextProcessor: {e}")
    
    # Проверка WebSearch
    try:
        from cogniflex.websearch.web_search_engine import WebSearchEngine
        print("   ✅ WebSearchEngine доступен")
    except Exception as e:
        print(f"   ❌ WebSearchEngine: {e}")
    
    # Проверка Monitoring
    try:
        from cogniflex.monitoring.system_monitor import SystemMonitor
        print("   ✅ SystemMonitor доступен")
    except Exception as e:
        print(f"   ❌ SystemMonitor: {e}")
    
    # Проверка Security
    try:
        from cogniflex.security import SecurityManager
        print("   ✅ SecurityManager доступен")
    except Exception as e:
        print(f"   ❌ SecurityManager: {e}")
    
    # 6. Проверка отсутствующих компонентов в ComponentInitializer
    print("\n6. 🔍 Поиск отсутствующих компонентов...")
    
    # Ищем Python файлы в директориях
    potential_components = []
    
    for root, dirs, files in os.walk(base_dir):
        # Пропускаем __pycache__ и тесты
        dirs[:] = [d for d in dirs if not d.startswith('__') and d != 'tests']
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                
                # Читаем файл для поиска классов
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Ищем определения классов
                    import re
                    class_matches = re.findall(r'class\s+(\w+)\s*\(', content)
                    
                    for class_name in class_matches:
                        # Пропускаем тестовые и внутренние классы
                        if not class_name.startswith('Test') and not class_name.startswith('_'):
                            potential_components.append({
                                'class': class_name,
                                'file': rel_path,
                                'registered': class_name.lower() in [f.lower() for f in factories]
                            })
                except:
                    pass
    
    # Показываем потенциальные компоненты
    registered_count = 0
    unregistered_count = 0
    
    print("\n   📋 Потенциальные компоненты:")
    for comp in sorted(potential_components, key=lambda x: x['file']):
        status = "✅" if comp['registered'] else "❌"
        print(f"   {status} {comp['class']} в {comp['file']}")
        
        if comp['registered']:
            registered_count += 1
        else:
            unregistered_count += 1
    
    print(f"\n   📊 Зарегистрировано: {registered_count}")
    print(f"   ❌ Не зарегистрировано: {unregistered_count}")
    
    # 7. Итоги
    print("\n🎉 Комплексная проверка завершена!")
    
    print("\n✅ Успешно интегрировано:")
    print(f"   🏗️ ComponentInitializer: {len(factories)} компонентов")
    print(f"   🧠 CoreBrain: {len(brain_components)} компонентов")
    print(f"   📁 Директорий: {len(existing_dirs)}/{len(expected_dirs)}")
    print(f"   📦 Потенциальных компонентов: {registered_count}/{len(potential_components)}")
    
    if unregistered_count > 0:
        print(f"\n⚠️ Требует внимания:")
        print(f"   ❌ Не зарегистрированных компонентов: {unregistered_count}")
        print(f"   ❌ Отсутствующих директорий: {len(missing_dirs)}")
    
    print("\n🚀 Рекомендации:")
    if unregistered_count > 0:
        print("   - Зарегистрировать отсутствующие компоненты в ComponentInitializer")
        print("   - Создать фабрики для новых компонентов")
        print("   - Настроить зависимости между компонентами")
    
    if len(missing_dirs) > 0:
        print("   - Создать отсутствующие директории")
        print("   - Реализовать базовую функциональность")
    
    print("   - Провести тестирование интеграции")
    print("   - Оптимизировать производительность")
    
    return True

if __name__ == "__main__":
    success = check_system_integration()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
CogniFlex Comprehensive Architecture Test
Расширенное тестирование всех компонентов и связей системы CogniFlex
"""

import sys
import os
import time
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

def test_system_initialization():
    """Тестирование полной инициализации системы."""
    print("=" * 80)
    print("🚀 COGNIFLEX COMPREHENSIVE ARCHITECTURE TEST")
    print("=" * 80)

    # Этап 1: Тестирование импортов всех модулей
    print("\n📦 ЭТАП 1: ТЕСТИРОВАНИЕ ИМПОРТОВ ВСЕХ МОДУЛЕЙ")

    modules_to_test = {
        'core_brain': 'cogniflex.core.core_brain',
        'component_initializer': 'cogniflex.core.component_initializer',
        'response_generator': 'cogniflex.core.response_generator',
        'memory_manager': 'cogniflex.memory.memory_manager',
        'knowledge_graph': 'cogniflex.knowledge.knowledge_graph',
        'ml_unit': 'cogniflex.mlearning.ml_unit',
        'chat_module': 'cogniflex.gui.chat_module',
        'core_gui': 'cogniflex.gui.core_gui',
        'text_processor': 'cogniflex.mlearning.unified_text_processor',
        'contradiction_core': 'cogniflex.contradiction.contradiction_core',
        'learning_manager': 'cogniflex.learning.learning_manager',
        'adaptation_manager': 'cogniflex.adaptation.adaptation_manager',
        'ethics_framework': 'cogniflex.ethics.ethics_framework',
        'neuromorphic_simulator': 'cogniflex.neuromorphic.neuromorphic_simulator',
        'web_search_engine': 'cogniflex.websearch.web_search_engine',
        'distributed_system': 'cogniflex.distributed.distributed_system'
    }

    import_results = {}
    for module_name, module_path in modules_to_test.items():
        try:
            print(f"  🔄 Импорт {module_name}...")
            __import__(module_path)
            import_results[module_name] = True
            print(f"  ✅ {module_name}: успешно импортирован")
        except ImportError as e:
            import_results[module_name] = f"ImportError: {e}"
            print(f"  ❌ {module_name}: ошибка импорта - {e}")
        except Exception as e:
            import_results[module_name] = f"Error: {e}"
            print(f"  ⚠️ {module_name}: неожиданная ошибка - {e}")

    # Этап 2: Тестирование создания CoreBrain
    print("\n📦 ЭТАП 2: ТЕСТИРОВАНИЕ СОЗДАНИЯ COREBRAIN")

    brain = None
    brain_creation_time = 0
    try:
        from cogniflex.core.core_brain import CoreBrain

        print("  🔄 Создание CoreBrain...")
        start_time = time.time()
        brain = CoreBrain()
        brain_creation_time = time.time() - start_time
        print(f"  ✅ CoreBrain создан за {brain_creation_time:.2f} сек")
        
        # Проверяем базовые атрибуты
        brain_attrs = {
            'components': hasattr(brain, 'components'),
            'cache_dir': hasattr(brain, 'cache_dir'),
            'memory_manager': hasattr(brain, 'memory_manager'),
            'query_logger': hasattr(brain, 'query_logger'),
            'events': hasattr(brain, 'events'),
            'running': hasattr(brain, 'running')
        }

        print("  🔧 Базовые атрибуты brain:")
        for attr, exists in brain_attrs.items():
            status = "✅" if exists else "❌"
            print(f"    {status} {attr}: {'найден' if exists else 'не найден'}")

    except Exception as e:
        print(f"  ❌ Ошибка создания CoreBrain: {e}")
        traceback.print_exc()

    # Этап 3: Тестирование component_initializer
    print("\n📦 ЭТАП 3: ТЕСТИРОВАНИЕ COMPONENT_INITIALIZER")

    component_init_results = {}
    if brain:
        try:
            from cogniflex.core.component_initializer import ComponentInitializer

            print("  🔄 Создание ComponentInitializer...")
            component_init = ComponentInitializer(brain)
            print("  ✅ ComponentInitializer создан")

            # Проверяем методы инициализатора
            init_methods = [
                'initialize_components',
                'initialize_gui',
                '_init_memory_manager',
                '_init_ml_unit',
                '_init_knowledge_graph',
                '_init_response_generator'
            ]

            print("  🔧 Методы ComponentInitializer:")
            for method in init_methods:
                exists = hasattr(component_init, method)
                status = "✅" if exists else "❌"
                print(f"    {status} {method}: {'доступен' if exists else 'недоступен'}")

            component_init_results['creation'] = True

        except Exception as e:
            component_init_results['creation'] = f"Error: {e}"
            print(f"  ❌ Ошибка ComponentInitializer: {e}")
            traceback.print_exc()
    else:
        component_init_results['creation'] = "Brain not available"

    # Этап 4: Тестирование GUI создания
    print("\n📦 ЭТАП 4: ТЕСТИРОВАНИЕ GUI СОЗДАНИЯ")

    gui_results = {}
    if brain:
        try:
            from cogniflex.gui.core_gui import create_gui

            print("  🔄 Создание GUI через create_gui...")
            gui = create_gui(brain=brain)
            print("  ✅ GUI создан")

            # Проверяем GUI атрибуты
            gui_attrs = {
                'brain': hasattr(gui, 'brain'),
                'root': hasattr(gui, 'root'),
                'content_area': hasattr(gui, 'content_area'),
                'colors': hasattr(gui, 'colors'),
                'chat_module': hasattr(gui, 'chat_module')
            }

            print("  🔧 Атрибуты GUI:")
            for attr, exists in gui_attrs.items():
                status = "✅" if exists else "❌"
                print(f"    {status} {attr}: {'инициализирован' if exists else 'не инициализирован'}")

            gui_results['creation'] = True
            gui_results['gui'] = gui

        except Exception as e:
            gui_results['creation'] = f"Error: {e}"
            print(f"  ❌ Ошибка создания GUI: {e}")
            traceback.print_exc()
    else:
        gui_results['creation'] = "Brain not available"

    # Этап 5: Тестирование модулей GUI
    print("\n📦 ЭТАП 5: ТЕСТИРОВАНИЕ МОДУЛЕЙ GUI")

    gui_modules_results = {}
    if gui_results.get('creation') and 'gui' in gui_results:
        gui = gui_results['gui']

        expected_modules = [
            'chat_module', 'analytics_module', 'knowledge_module',
            'contradiction_module', 'memory_module', 'learning_module',
            'settings_module', 'neuromorphic_module'
        ]

        print("  🔧 Проверка модулей GUI:")
        for module_name in expected_modules:
            exists = hasattr(gui, module_name)
            is_not_none = exists and getattr(gui, module_name) is not None

            status = "✅" if is_not_none else "⚠️" if exists else "❌"
            print(f"    {status} {module_name}: {'инициализирован' if is_not_none else 'не найден' if not exists else 'None'}")

            gui_modules_results[module_name] = is_not_none

        # Особая проверка chat_module
        if hasattr(gui, 'chat_module') and gui.chat_module:
            chat = gui.chat_module
            print("  🔧 Детальная проверка chat_module:")

            chat_methods = [
                'activate', 'deactivate', '_send_message', '_add_message',
                'gui', 'message_history', 'input_text', 'send_button'
            ]

            for method in chat_methods:
                exists = hasattr(chat, method)
                status = "✅" if exists else "❌"
                print(f"    {status} {method}: {'доступен' if exists else 'недоступен'}")

    # Этап 6: Тестирование связей между компонентами
    print("\n📦 ЭТАП 6: ТЕСТИРОВАНИЕ СВЯЗЕЙ МЕЖДУ КОМПОНЕНТАМИ")

    connections_results = {}

    if brain:
        # Тест 1: Brain ↔ Memory Manager
        print("  🔗 Brain ↔ Memory Manager:")
        brain_memory_ok = (
            hasattr(brain, 'memory_manager') and
            brain.memory_manager is not None and
            hasattr(brain.memory_manager, 'get_memory_statistics')
        )
        connections_results['brain_memory'] = brain_memory_ok
        print(f"    {'✅' if brain_memory_ok else '❌'} Связь: {'работает' if brain_memory_ok else 'не работает'}")

        # Тест 2: Brain ↔ Components
        print("  🔗 Brain ↔ Components:")
        brain_components_ok = (
            hasattr(brain, 'components') and
            isinstance(brain.components, dict) and
            len(brain.components) > 0
        )
        connections_results['brain_components'] = brain_components_ok
        print(f"    {'✅' if brain_components_ok else '❌'} Компоненты: {'инициализированы' if brain_components_ok else 'не инициализированы'}")

        # Тест 3: Brain ↔ Process Query
        print("  🔗 Brain ↔ Process Query:")
        brain_query_ok = hasattr(brain, 'process_query')
        connections_results['brain_query'] = brain_query_ok
        print(f"    {'✅' if brain_query_ok else '❌'} Process Query: {'доступен' if brain_query_ok else 'недоступен'}")

    if gui_results.get('creation') and 'gui' in gui_results:
        gui = gui_results['gui']

        # Тест 4: GUI ↔ Brain
        print("  🔗 GUI ↔ Brain:")
        gui_brain_ok = (
            hasattr(gui, 'brain') and
            gui.brain is not None and
            gui.brain is brain
        )
        connections_results['gui_brain'] = gui_brain_ok
        print(f"    {'✅' if gui_brain_ok else '❌'} Связь: {'корректна' if gui_brain_ok else 'неверна'}")

        # Тест 5: GUI ↔ Chat Module
        print("  🔗 GUI ↔ Chat Module:")
        gui_chat_ok = (
            hasattr(gui, 'chat_module') and
            gui.chat_module is not None
        )
        connections_results['gui_chat'] = gui_chat_ok
        print(f"    {'✅' if gui_chat_ok else '❌'} Chat Module: {'инициализирован' if gui_chat_ok else 'не инициализирован'}")

        # Тест 6: Chat ↔ Brain через GUI
        if gui_chat_ok:
            print("  🔗 Chat ↔ Brain через GUI:")
            chat_brain_via_gui_ok = (
                hasattr(gui.chat_module, 'gui') and
                hasattr(gui.chat_module.gui, 'brain') and
                gui.chat_module.gui.brain is brain
            )
            connections_results['chat_brain_via_gui'] = chat_brain_via_gui_ok
            print(f"    {'✅' if chat_brain_via_gui_ok else '❌'} Связь: {'работает' if chat_brain_via_gui_ok else 'не работает'}")

    # Этап 7: Тестирование функциональности
    print("\n📦 ЭТАП 7: ТЕСТИРОВАНИЕ ФУНКЦИОНАЛЬНОСТИ")

    functionality_results = {}

    # Тест функциональности Brain
    if brain:
        print("  🔧 Функциональность Brain:")

        # Test get_system_status
        try:
            status = brain.get_system_status()
            functionality_results['brain_status'] = f"Статус: {status}"
            print(f"    ✅ get_system_status: {status}")
        except Exception as e:
            functionality_results['brain_status'] = f"Ошибка: {e}"
            print(f"    ❌ get_system_status: {e}")

        # Test process_query (если доступен)
        if hasattr(brain, 'process_query'):
            try:
                test_result = brain.process_query("Тестовый запрос")
                functionality_results['brain_query'] = "Работает"
                print(f"    ✅ process_query: работает (тип: {type(test_result).__name__})")
            except Exception as e:
                functionality_results['brain_query'] = f"Ошибка: {e}"
                print(f"    ❌ process_query: {e}")

    # Тест функциональности Memory Manager
    if brain and hasattr(brain, 'memory_manager') and brain.memory_manager:
        print("  🔧 Функциональность Memory Manager:")

        try:
            stats = brain.memory_manager.get_memory_statistics()
            functionality_results['memory_stats'] = f"{len(stats) if stats else 0} метрик"
            print(f"    ✅ get_memory_statistics: {len(stats) if stats else 0} метрик")
        except Exception as e:
            functionality_results['memory_stats'] = f"Ошибка: {e}"
            print(f"    ❌ get_memory_statistics: {e}")

    # Этап 8: ИТОГИ ТЕСТИРОВАНИЯ
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ")
    print("=" * 80)

    # Сводка по имортам
    successful_imports = sum(1 for result in import_results.values() if result is True)
    total_imports = len(import_results)
    print(f"📦 ИМПОРТЫ: {successful_imports}/{total_imports} модулей успешно импортированы")

    # Сводка по компонентам
    components_summary = {
        'Brain': brain is not None,
        'Component Initializer': component_init_results.get('creation') is True,
        'GUI': gui_results.get('creation') is True,
        'GUI Modules': sum(gui_modules_results.values()) if gui_modules_results else 0
    }

    print("🏗️ КОМПОНЕНТЫ:")
    for component, status in components_summary.items():
        print(f"  {'✅' if status else '❌'} {component}: {'работает' if status else 'не работает'}")

    # Сводка по связям
    connections_summary = sum(1 for result in connections_results.values() if result)
    total_connections = len(connections_results)
    print(f"🔗 СВЯЗИ: {connections_summary}/{total_connections} связей работают корректно")

    # Общий результат
    overall_success = (
        successful_imports >= total_imports * 0.8 and  # 80% модулей импортированы
        all(components_summary.values()) and           # Все основные компоненты работают
        connections_summary >= total_connections * 0.7  # 70% связей работают
    )

    print(f"\n{'🎉' if overall_success else '⚠️'} ОБЩИЙ РЕЗУЛЬТАТ: {'СИСТЕМА ГОТОВА К РАБОТЕ' if overall_success else 'ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ'}")

    # Детальная диагностика
    print("\n🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА:")

    if not overall_success:
        print("\n❌ ПРОБЛЕМЫ:")

        if successful_imports < total_imports * 0.8:
            failed_imports = [name for name, result in import_results.items() if result is not True]
            print(f"  • Не удалось импортировать: {', '.join(failed_imports)}")

        failed_components = [name for name, status in components_summary.items() if not status]
        if failed_components:
            print(f"  • Проблемные компоненты: {', '.join(failed_components)}")

        failed_connections = [name for name, result in connections_results.items() if not result]
        if failed_connections:
            print(f"  • Проблемные связи: {', '.join(failed_connections)}")

    print("\n✅ РАБОТАЮЩИЕ КОМПОНЕНТЫ:")
    if brain:
        print(f"  • Brain: инициализирован за {brain_creation_time:.2f} сек")
    if gui_results.get('creation'):
        print(f"  • GUI: инициализирован с {sum(gui_modules_results.values())} модулями")
    if brain and hasattr(brain, 'memory_manager'):
        print("  • Memory Manager: подключен")

    print("\n📈 ПРОИЗВОДИТЕЛЬНОСТЬ:")
    if brain_creation_time > 0:
        print(f"  • Время инициализации: {brain_creation_time:.2f} сек")
    
    return {
        'imports': import_results,
        'components': components_summary,
        'connections': connections_results,
        'functionality': functionality_results,
        'overall_success': overall_success
    }

if __name__ == "__main__":
    results = test_system_initialization()

    # Сохраняем результаты в файл для анализа
    with open('comprehensive_test_results.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n💾 Результаты сохранены в: comprehensive_test_results.json")

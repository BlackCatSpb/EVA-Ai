#!/usr/bin/env python3
"""
CogniFlex Extended Architecture Test
Расширенное тестирование связей chat_module со всей архитектурой проекта
"""

import sys
import os
import time
import traceback
from pathlib import Path

def test_brain_full_initialization():
    """Расширенное тестирование инициализации brain."""
    print("🔍 Расширенное тестирование инициализации brain...")

    try:
        # Импортируем и создаем CoreBrain
        sys.path.insert(0, str(Path(__file__).parent))
        from cogniflex.core.core_brain import CoreBrain

        print("  📦 Создание CoreBrain...")
        start_time = time.time()
        brain = CoreBrain()
        init_time = time.time() - start_time
        print(f"Время инициализации: {init_time:.2f} сек")
        # Проверяем базовые атрибуты
        basic_attrs = {
            'cache_dir': hasattr(brain, 'cache_dir'),
            'components': hasattr(brain, 'components'),
            'running': hasattr(brain, 'running'),
            'brain_ref': brain.brain is brain,
            'query_logger': hasattr(brain, 'query_logger'),
        }

        print("  🔧 Базовые атрибуты brain:")
        for attr, exists in basic_attrs.items():
            status = "✅" if exists else "❌"
            print(f"    {status} {attr}: {'найден' if exists else 'не найден'}")

        # Проверяем компоненты
        components_check = {
            'memory_manager': hasattr(brain, 'memory_manager') and brain.memory_manager is not None,
            'components_dict': hasattr(brain, 'components') and isinstance(brain.components, dict),
            'process_query': hasattr(brain, 'process_query'),
            'running_state': hasattr(brain, 'running'),
        }

        print("  🔧 Компоненты brain:")
        for comp, exists in components_check.items():
            status = "✅" if exists else "❌"
            print(f"    {status} {comp}: {'инициализирован' if exists else 'не инициализирован'}")

        # Проверяем методы
        methods_check = {
            'get_system_status': hasattr(brain, 'get_system_status'),
            'get_system_dashboard_data': hasattr(brain, 'get_system_dashboard_data'),
            'get_resource_snapshot': hasattr(brain, 'get_resource_snapshot'),
            'get_cache_stats': hasattr(brain, 'get_cache_stats'),
        }

        print("  🔧 Методы brain:")
        for method, exists in methods_check.items():
            status = "✅" if exists else "❌"
            print(f"    {status} {method}: {'доступен' if exists else 'недоступен'}")

        # Тестируем вызовы методов
        print("  🔧 Тестирование вызовов методов:")
        try:
            status = brain.get_system_status()
            print(f"    ✅ get_system_status: {status}")
        except Exception as e:
            print(f"    ❌ get_system_status: ошибка {e}")

        try:
            dashboard = brain.get_system_dashboard_data()
            print(f"    ✅ get_system_dashboard_data: {len(dashboard) if dashboard else 0} метрик")
        except Exception as e:
            print(f"    ❌ get_system_dashboard_data: ошибка {e}")

        return brain, {**basic_attrs, **components_check, **methods_check}

    except Exception as e:
        print(f"  ❌ Критическая ошибка инициализации brain: {e}")
        traceback.print_exc()
        return None, {}

def test_gui_initialization_chain(brain):
    """Тестирование цепочки инициализации GUI."""
    print("\n🔍 Тестирование цепочки инициализации GUI...")

    if not brain:
        print("  ❌ Brain недоступен для тестирования GUI")
        return False

    gui = None
    try:
        # Тест 1: Импорт GUI
        print("  📦 Шаг 1: Импорт GUI модулей...")
        from cogniflex.gui.core_gui import create_gui, CogniFlexGUI
        print("  ✅ GUI модули импортированы")

        # Тест 2: Создание GUI через create_gui
        print("  📦 Шаг 2: Создание GUI через create_gui...")
        gui = create_gui(brain=brain)
        print("  ✅ GUI создан через create_gui")

        # Тест 3: Проверка базовых атрибутов GUI
        print("  📦 Шаг 3: Проверка атрибутов GUI...")
        gui_attrs = {
            'brain': hasattr(gui, 'brain') and gui.brain is not None,
            'brain_correct': gui.brain is brain if hasattr(gui, 'brain') else False,
            'content_area': hasattr(gui, 'content_area'),
            'root': hasattr(gui, 'root'),
            'colors': hasattr(gui, 'colors'),
            'gui_queue': hasattr(gui, 'gui_queue'),
        }

        for attr, exists in gui_attrs.items():
            status = "✅" if exists else "❌"
            print(f"    {status} {attr}: {'инициализирован' if exists else 'не инициализирован'}")

        # Тест 4: Проверка методов GUI
        print("  📦 Шаг 4: Проверка методов GUI...")
        gui_methods = {
            '_init_modules': hasattr(gui, '_init_modules'),
            '_switch_view': hasattr(gui, '_switch_view'),
            '_create_interface': hasattr(gui, '_create_interface'),
            'start': hasattr(gui, 'start'),
        }

        for method, exists in gui_methods.items():
            status = "✅" if exists else "❌"
            print(f"    {status} {method}: {'доступен' if exists else 'недоступен'}")

        return gui

    except Exception as e:
        print(f"  ❌ Ошибка инициализации GUI: {e}")
        traceback.print_exc()
        return None

def test_gui_module_initialization(gui):
    """Тестирование инициализации модулей GUI."""
    print("\n🔍 Тестирование инициализации модулей GUI...")

    if not gui:
        print("  ❌ GUI недоступен для тестирования модулей")
        return False

    try:
        # Тест 1: Проверка наличия content_area
        if not hasattr(gui, 'content_area') or gui.content_area is None:
            print("  ❌ content_area не инициализирован - модули не могут быть созданы")

            # Создаем content_area для тестирования
            print("  📦 Создание content_area для тестирования...")
            try:
                import tkinter as tk
                from tkinter import ttk

                # Создаем root окно
                gui.root = tk.Tk()
                gui.root.title("Test Window")
                gui.root.geometry("800x600")

                # Создаем main_container и content_area
                gui.main_container = ttk.Frame(gui.root)
                gui.main_container.pack(fill=tk.BOTH, expand=True)

                gui.content_frame = ttk.Frame(gui.main_container)
                gui.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

                gui.content_area = ttk.Frame(gui.content_frame)
                gui.content_area.pack(fill=tk.BOTH, expand=True)

                print("  ✅ content_area создан для тестирования")

            except Exception as e:
                print(f"  ❌ Не удалось создать content_area: {e}")
                return False

        # Тест 2: Вызов _init_modules
        print("  📦 Вызов _init_modules...")
        try:
            gui._init_modules()
            print("  ✅ _init_modules выполнен")
        except Exception as e:
            print(f"  ❌ Ошибка в _init_modules: {e}")
            traceback.print_exc()
            return False

        # Тест 3: Проверка инициализированных модулей
        print("  📦 Проверка инициализированных модулей...")
        expected_modules = ['chat_module', 'analytics_module', 'knowledge_module',
                          'contradiction_module', 'memory_module', 'learning_module',
                          'settings_module', 'neuromorphic_module']

        modules_status = {}
        for module_name in expected_modules:
            exists = hasattr(gui, module_name)
            is_not_none = exists and getattr(gui, module_name) is not None
            modules_status[module_name] = exists and is_not_none

            status = "✅" if is_not_none else "⚠️" if exists else "❌"
            print(f"    {status} {module_name}: {'инициализирован' if is_not_none else 'не найден' if not exists else 'None'}")

        # Особая проверка chat_module
        chat_module_ok = modules_status.get('chat_module', False)
        if chat_module_ok:
            chat = gui.chat_module
            print("  🔧 Детальная проверка chat_module:")

            # Проверяем атрибуты chat_module
            chat_attrs = {
                'gui': hasattr(chat, 'gui'),
                'message_history': hasattr(chat, 'message_history'),
                'input_text': hasattr(chat, 'input_text'),
                'send_button': hasattr(chat, 'send_button'),
            }

            for attr, exists in chat_attrs.items():
                status = "✅" if exists else "❌"
                print(f"    {status} {attr}: {'доступен' if exists else 'недоступен'}")

            # Проверяем методы chat_module
            chat_methods = {
                'activate': hasattr(chat, 'activate'),
                'deactivate': hasattr(chat, 'deactivate'),
                '_send_message': hasattr(chat, '_send_message'),
            }

            for method, exists in chat_methods.items():
                status = "✅" if exists else "❌"
                print(f"    {status} {method}: {'доступен' if exists else 'недоступен'}")

        return chat_module_ok

    except Exception as e:
        print(f"  ❌ Критическая ошибка тестирования модулей: {e}")
        traceback.print_exc()
        return False

def test_chat_brain_integration(chat_module, brain):
    """Тестирование интеграции chat_module с brain."""
    print("\n🔍 Тестирование интеграции chat_module с brain...")

    if not chat_module or not brain:
        print("  ❌ chat_module или brain недоступны для тестирования")
        return False

    try:
        # Тест 1: Проверка доступа chat к brain
        print("  📦 Проверка доступа chat_module к brain...")
        chat_has_brain = hasattr(chat_module, 'gui') and hasattr(chat_module.gui, 'brain')
        brain_accessible = chat_has_brain and chat_module.gui.brain is not None
        brain_correct = brain_accessible and chat_module.gui.brain is brain

        print(f"    {'✅' if chat_has_brain else '❌'} Chat имеет доступ к brain: {chat_has_brain}")
        print(f"    {'✅' if brain_accessible else '❌'} Brain доступен: {brain_accessible}")
        print(f"    {'✅' if brain_correct else '❌'} Brain корректный: {brain_correct}")

        # Тест 2: Проверка компонентов brain из chat
        if brain_accessible:
            print("  📦 Проверка компонентов brain из chat...")

            brain_components = {
                'memory_manager': hasattr(brain, 'memory_manager'),
                'process_query': hasattr(brain, 'process_query'),
                'components': hasattr(brain, 'components'),
            }

            for comp, exists in brain_components.items():
                status = "✅" if exists else "❌"
                value = getattr(brain, comp, None)
                print(f"    {status} {comp}: {'найден' if exists else 'не найден'} ({type(value).__name__ if value else 'None'})")

        # Тест 3: Проверка методов обработки
        if hasattr(brain, 'process_query'):
            print("  📦 Тестирование метода process_query...")
            try:
                # Пробуем вызвать с тестовым запросом
                test_query = "Тестовый запрос для проверки системы"
                result = brain.process_query(test_query)
                print(f"    ✅ process_query работает: {type(result).__name__}")
                print(f"    📝 Результат: {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
            except Exception as e:
                print(f"    ❌ process_query ошибка: {e}")

        return chat_has_brain and brain_accessible and brain_correct

    except Exception as e:
        print(f"  ❌ Ошибка тестирования интеграции: {e}")
        traceback.print_exc()
        return False

def test_memory_manager_integration(brain):
    """Тестирование интеграции memory_manager."""
    print("\n🔍 Тестирование интеграции memory_manager...")

    if not brain:
        print("  ❌ Brain недоступен для тестирования memory_manager")
        return False

    try:
        # Тест 1: Проверка наличия memory_manager
        has_memory_manager = hasattr(brain, 'memory_manager') and brain.memory_manager is not None
        print(f"    {'✅' if has_memory_manager else '❌'} memory_manager: {'найден' if has_memory_manager else 'не найден'}")

        if not has_memory_manager:
            return False

        memory_manager = brain.memory_manager

        # Тест 2: Проверка методов memory_manager
        memory_methods = {
            'get_memory_statistics': hasattr(memory_manager, 'get_memory_statistics'),
            'analyze_memory_usage': hasattr(memory_manager, 'analyze_memory_usage'),
            'get_all_nodes': hasattr(memory_manager, 'get_all_nodes'),
            'initialized': hasattr(memory_manager, 'initialized'),
        }

        print("  📦 Методы memory_manager:")
        for method, exists in memory_methods.items():
            status = "✅" if exists else "❌"
            print(f"    {status} {method}: {'доступен' if exists else 'недоступен'}")

        # Тест 3: Проверка состояния инициализации
        if hasattr(memory_manager, 'initialized'):
            initialized = memory_manager.initialized
            print(f"    {'✅' if initialized else '❌'} initialized: {initialized}")

        # Тест 4: Вызов методов
        if hasattr(memory_manager, 'get_memory_statistics'):
            try:
                stats = memory_manager.get_memory_statistics()
                print(f"    ✅ get_memory_statistics: {len(stats) if stats else 0} метрик")
            except Exception as e:
                print(f"    ❌ get_memory_statistics ошибка: {e}")

        return has_memory_manager

    except Exception as e:
        print(f"  ❌ Ошибка тестирования memory_manager: {e}")
        traceback.print_exc()
        return False

def main():
    """Основная функция расширенного тестирования."""
    print("=" * 70)
    print("🔬 COGNIFLEX EXTENDED ARCHITECTURE TEST")
    print("=" * 70)
    print("Расширенное тестирование связей chat_module со всей архитектурой")
    print()

    # Этап 1: Тестирование brain
    brain, brain_status = test_brain_full_initialization()

    # Этап 2: Тестирование GUI
    gui = test_gui_initialization_chain(brain)

    # Этап 3: Тестирование модулей GUI
    modules_ok = test_gui_module_initialization(gui)

    # Этап 4: Тестирование интеграции chat с brain
    if gui and hasattr(gui, 'chat_module') and gui.chat_module:
        chat_integration_ok = test_chat_brain_integration(gui.chat_module, brain)
    else:
        chat_integration_ok = False
        print("\n❌ Пропуск тестирования chat-brain интеграции - chat_module не инициализирован")

    # Этап 5: Тестирование memory_manager
    memory_ok = test_memory_manager_integration(brain)

    # Итоги
    print("\n" + "=" * 70)
    print("📊 РАСШИРЕННЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 70)

    results = {
        'Brain инициализация': brain is not None,
        'GUI создание': gui is not None,
        'Модули GUI': modules_ok,
        'Chat-Brain интеграция': chat_integration_ok,
        'Memory Manager': memory_ok,
    }

    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}: {'ПРОЙДЕН' if passed else 'ПРОВАЛЕН'}")

    overall_success = all(results.values())
    print(f"\n{'🎉' if overall_success else '⚠️'} ОБЩИЙ РЕЗУЛЬТАТ: {'ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if overall_success else 'ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ'}")

    if not overall_success:
        print("\n🔧 ПРОБЛЕМЫ ДЛЯ ИСПРАВЛЕНИЯ:")
        failed_tests = [name for name, passed in results.items() if not passed]
        for i, test in enumerate(failed_tests, 1):
            print(f"  {i}. {test}")

    print("\n🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА:")
    if brain:
        print(f"  • Brain компонентов: {len(brain.components) if hasattr(brain, 'components') else 0}")
        print(f"  • Memory manager: {'инициализирован' if hasattr(brain, 'memory_manager') and brain.memory_manager else 'не инициализирован'}")
    else:
        print("  • Brain: не создан")

    if gui:
        modules_count = sum(1 for attr in dir(gui) if attr.endswith('_module') and getattr(gui, attr) is not None)
        print(f"  • GUI модулей: {modules_count}")
        print(f"  • Chat module: {'инициализирован' if hasattr(gui, 'chat_module') and gui.chat_module else 'не инициализирован'}")
    else:
        print("  • GUI: не создан")

    print("=" * 70)

if __name__ == "__main__":
    main()

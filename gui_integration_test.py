#!/usr/bin/env python3
"""
CogniFlex GUI Integration Test
Проверяет интеграцию GUI с ядром системы
"""

import sys
import os
import time
from pathlib import Path

def test_brain_initialization():
    """Тестирует инициализацию brain компонентов."""
    print("🔍 Тестирование инициализации brain...")

    try:
        # Импортируем core_brain
        sys.path.insert(0, str(Path(__file__).parent))
        from cogniflex.core.core_brain import CoreBrain

        # Создаем экземпляр
        print("  📦 Создание CoreBrain...")
        brain = CoreBrain()

        # Проверяем базовые атрибуты
        print("  ✅ CoreBrain создан успешно")

        # Проверяем наличие основных компонентов
        components_check = {
            'memory_manager': hasattr(brain, 'memory_manager'),
            'components': hasattr(brain, 'components'),
            'process_query': hasattr(brain, 'process_query'),
            'running': hasattr(brain, 'running'),
        }

        print("  🔧 Проверка компонентов:")
        for component, exists in components_check.items():
            status = "✅" if exists else "❌"
            print(f"    {status} {component}: {'найден' if exists else 'не найден'}")

        return brain, components_check

    except Exception as e:
        print(f"  ❌ Ошибка инициализации brain: {e}")
        return None, {}

def test_gui_creation(brain):
    """Тестирует создание GUI с brain."""
    print("\n🔍 Тестирование создания GUI...")

    if not brain:
        print("  ❌ Brain недоступен для тестирования GUI")
        return False

    try:
        from cogniflex.gui.core_gui import create_gui

        print("  📦 Создание GUI...")
        gui = create_gui(brain=brain)

        # Проверяем, что GUI получил brain
        gui_has_brain = hasattr(gui, 'brain') and gui.brain is not None
        brain_correct = gui.brain is brain

        print("  🔧 Проверка GUI:")
        print(f"    {'✅' if gui_has_brain else '❌'} GUI имеет brain: {gui_has_brain}")
        print(f"    {'✅' if brain_correct else '❌'} Brain корректный: {brain_correct}")

        return gui_has_brain and brain_correct

    except Exception as e:
        print(f"  ❌ Ошибка создания GUI: {e}")
        return False

def test_chat_module(gui):
    """Тестирует chat модуль."""
    print("\n🔍 Тестирование chat модуля...")

    if not gui:
        print("  ❌ GUI недоступен для тестирования chat")
        return False

    try:
        # Проверяем наличие chat_module
        has_chat_module = hasattr(gui, 'chat_module')

        print("  🔧 Проверка chat модуля:")
        print(f"    {'✅' if has_chat_module else '❌'} Chat модуль: {'найден' if has_chat_module else 'не найден'}")

        if has_chat_module:
            # Проверяем, что chat модуль имеет доступ к brain
            chat_has_brain = hasattr(gui.chat_module, 'gui') and hasattr(gui.chat_module.gui, 'brain')
            brain_accessible = chat_has_brain and gui.chat_module.gui.brain is not None

            print(f"    {'✅' if chat_has_brain else '❌'} Chat имеет доступ к brain: {chat_has_brain}")
            print(f"    {'✅' if brain_accessible else '❌'} Brain доступен из chat: {brain_accessible}")

            return chat_has_brain and brain_accessible

        return False

    except Exception as e:
        print(f"  ❌ Ошибка тестирования chat модуля: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("🧪 COGNIFLEX GUI INTEGRATION TEST")
    print("=" * 60)
    print("Тестирование интеграции GUI с ядром системы")
    print()

    # Тест 1: Инициализация brain
    brain, components = test_brain_initialization()

    # Тест 2: Создание GUI
    gui_success = test_gui_creation(brain)

    # Тест 3: Chat модуль
    chat_success = test_chat_module(brain)

    # Итоги
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)

    all_components_found = all(components.values())
    print(f"{'✅' if all_components_found else '❌'} Компоненты brain: {'все найдены' if all_components_found else 'некоторые отсутствуют'}")
    print(f"{'✅' if gui_success else '❌'} GUI интеграция: {'успешна' if gui_success else 'провалена'}")
    print(f"{'✅' if chat_success else '❌'} Chat модуль: {'работает' if chat_success else 'не работает'}")

    overall_success = all_components_found and gui_success and chat_success
    print(f"\n{'🎉' if overall_success else '⚠️'} ОБЩИЙ РЕЗУЛЬТАТ: {'ВСЕ СИСТЕМЫ РАБОТАЮТ' if overall_success else 'ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ'}")

    if not overall_success:
        print("\n🔧 РЕКОМЕНДАЦИИ:")
        if not all_components_found:
            print("  - Проверить инициализацию компонентов brain")
        if not gui_success:
            print("  - Проверить передачу brain в GUI")
        if not chat_success:
            print("  - Проверить доступ chat модуля к brain")

    print("=" * 60)

if __name__ == "__main__":
    main()

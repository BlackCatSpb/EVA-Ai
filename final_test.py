#!/usr/bin/env python3
"""
Финальный тест исправлений CogniFlex
"""

import sys
import os
import traceback

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def test_imports():
    """Тестирует критические импорты."""
    try:
        from cogniflex.core.core_brain import CoreBrain
        from cogniflex.mlearning.ml_unit import MLUnit
        from cogniflex.memory.hybrid_token_cache import HybridTokenCache
        print("✅ Все критические импорты работают")
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        return False

def test_brain_creation():
    """Тестирует создание CoreBrain."""
    try:
        from cogniflex.core.core_brain import CoreBrain
        brain = CoreBrain()
        print("✅ CoreBrain создан успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания CoreBrain: {e}")
        return False

def test_ml_unit_creation():
    """Тестирует создание MLUnit."""
    try:
        class MockBrain:
            def __init__(self):
                self.cache_dir = "."
                self.config = {}
                self.components = {}
        
        from cogniflex.mlearning.ml_unit import MLUnit
        brain = MockBrain()
        ml_unit = MLUnit(brain)
        print("✅ MLUnit создан успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания MLUnit: {e}")
        return False

def main():
    print("🔧 ФИНАЛЬНЫЙ ТЕСТ ИСПРАВЛЕНИЙ")
    print("=" * 40)
    
    tests = [
        ("Импорты", test_imports),
        ("CoreBrain", test_brain_creation), 
        ("MLUnit", test_ml_unit_creation)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Критическая ошибка в {name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("📊 РЕЗУЛЬТАТЫ ИСПРАВЛЕНИЙ:")
    for i, (name, _) in enumerate(tests):
        status = "✅ ИСПРАВЛЕНО" if results[i] else "❌ ТРЕБУЕТ ВНИМАНИЯ"
        print(f"{name}: {status}")
    
    success_rate = sum(results) / len(results)
    print(f"\nОбщий результат: {sum(results)}/{len(results)} ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        print("🎉 Критические исправления выполнены! Система готова к запуску.")
    else:
        print("⚠️ Требуются дополнительные исправления.")

if __name__ == "__main__":
    main()

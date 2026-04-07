#!/usr/bin/env python3
"""
Тест генерации текста для ЕВА
Проверяет качество генерируемых ответов
"""

import os
import sys
import time

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def test_individual_modules():
    """Тестирует инициализацию отдельных модулей."""
    print("=== Тест инициализации отдельных модулей ===")
    
    modules_to_test = [
        ("ConfigManager", "cogniflex.core.config_manager", "ConfigManager"),
        ("SystemStateManager", "cogniflex.core.system_state", "SystemStateManager"),
        ("ResourceManager", "cogniflex.core.resource_manager", "ResourceManager"),
        ("SystemMetricsManager", "cogniflex.core.system_metrics", "SystemMetricsManager"),
        ("ComponentInitializer", "cogniflex.core.component_initializer", "ComponentInitializer"),
    ]
    
    results = {}
    
    for name, module_path, class_name in modules_to_test:
        try:
            print(f"\nТестирование {name}...")
            
            # Импортируем модуль
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            # Создаем экземпляр с минимальными параметрами
            if name == "ComponentInitializer":
                # Создаем заглушку brain
                class MockBrain:
                    def __init__(self):
                        self.components = {}
                        self.cache_dir = "test_cache"
                        os.makedirs(self.cache_dir, exist_ok=True)
                
                instance = cls(MockBrain())
            elif name == "ResourceManager":
                instance = cls()
            else:
                instance = cls()
            
            print(f"✅ {name} успешно инициализирован")
            results[name] = True
            
            # Тестируем базовые методы
            if hasattr(instance, 'get_status'):
                status = instance.get_status()
                print(f"  Статус: {type(status)}")
            
            if hasattr(instance, 'get_system_info'):
                info = instance.get_system_info()
                print(f"  Системная информация: {type(info)}")
                
        except Exception as e:
            print(f"❌ Ошибка в {name}: {e}")
            results[name] = False
    
    return results

def test_brain_initialization():
    """Тестирует инициализацию ядра системы."""
    print("\n=== Тест инициализации CoreBrain ===")
    
    try:
        from eva_ai.core.core_brain import CoreBrain
        
        # Создаем минимальную конфигурацию
        config = {
            "test_mode": True,
            "log_level": "INFO"
        }
        
        print("Создание экземпляра CoreBrain...")
        brain = CoreBrain(config)
        
        print("Проверка новых менеджеров...")
        if brain.config_manager:
            print("✅ ConfigManager инициализирован")
        if brain.state_manager:
            print("✅ SystemStateManager инициализирован")
        if brain.resource_manager:
            print("✅ ResourceManager инициализирован")
        
        print("Инициализация компонентов...")
        success = brain.initialize()
        
        if success:
            print("✅ Инициализация успешна!")
            
            # Получаем расширенный статус
            status = brain.get_status()
            print(f"Компонентов инициализировано: {status.get('components', 0)}")
            
            if 'system_state' in status:
                print(f"Состояние системы: {status['system_state'].get('current_state', 'unknown')}")
            
            if 'resources' in status:
                print(f"Статус ресурсов: {status['resources'].get('status', 'unknown')}")
            
            # Останавливаем систему
            brain.stop()
            print("✅ Система корректно остановлена")
            return True
            
        else:
            print("❌ Ошибка инициализации")
            return False
            
    except Exception as e:
        print(f"❌ Исключение: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_circular_dependencies():
    """Проверяет отсутствие циклических зависимостей."""
    print("\n=== Тест циклических зависимостей ===")
    
    try:
        # Тестируем импорты в правильном порядке
        print("Импорт core модулей...")
        from eva_ai.core.config_manager import ConfigManager
        from eva_ai.core.system_state import SystemStateManager
        from eva_ai.core.resource_manager import ResourceManager
        print("✅ Core модули импортированы без циклических зависимостей")
        
        print("Импорт mlearning модулей...")
        from eva_ai.mlearning.ml_unit import MLUnit
        print("✅ MLUnit импортирован без циклических зависимостей")
        
        print("Импорт memory модулей...")
        from eva_ai.memory.hybrid_token_cache import HybridTokenCache
        print("✅ HybridTokenCache импортирован без циклических зависимостей")
        
        return True
        
    except Exception as e:
        print(f"❌ Обнаружена циклическая зависимость: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов модулей ЕВА")
    print("=" * 50)
    
    # Тест отдельных модулей
    module_results = test_individual_modules()
    
    # Тест циклических зависимостей
    circular_test = test_circular_dependencies()
    
    # Тест инициализации brain
    brain_test = test_brain_initialization()
    
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 50)
    
    print("Модули:")
    for module, result in module_results.items():
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"  {module:20} : {status}")
    
    print(f"\nЦиклические зависимости: {'✅ ОТСУТСТВУЮТ' if circular_test else '❌ ОБНАРУЖЕНЫ'}")
    print(f"Инициализация CoreBrain: {'✅ УСПЕШНА' if brain_test else '❌ ПРОВАЛЕНА'}")
    
    total_passed = sum(module_results.values()) + circular_test + brain_test
    total_tests = len(module_results) + 2
    
    print(f"\nОбщий результат: {total_passed}/{total_tests} тестов пройдено ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        print("🎉 Все тесты успешно пройдены!")
    else:
        print("⚠️  Некоторые тесты провалены. Требуется доработка.")

def is_corrupted_text(text):
    """Проверяет, является ли текст поврежденным."""
    if not text or len(text.strip()) < 3:
        return True
    
    # Проверяем на чрезмерное повторение
    words = text.split()
    if len(words) > 3:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        for word, count in word_counts.items():
            if count > len(words) * 0.4:  # Более 40% повторений
                return True
    
    # Проверяем на странные символы
    strange_chars = 0
    for char in text:
        if ord(char) > 1200 or (ord(char) < 32 and char not in '\n\t'):
            strange_chars += 1
    
    if strange_chars > len(text) * 0.1:  # Более 10% странных символов
        return True
    
    return False

if __name__ == "__main__":
    success = test_text_generation()
    sys.exit(0 if success else 1)
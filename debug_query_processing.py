"""
Отладка обработки запросов в CogniFlex
Анализирует проблемы с query processing
"""

import os
import sys
import traceback
import time

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def test_core_brain_query():
    """Тестирует обработку запроса через CoreBrain."""
    print("=== Тест обработки запроса через CoreBrain ===")
    
    try:
        from cogniflex.core.core_brain import CoreBrain
        
        print("Создание CoreBrain...")
        brain = CoreBrain()
        
        print("Инициализация CoreBrain...")
        if not brain.initialize():
            print("❌ Инициализация провалена")
            return False
        
        print("✅ CoreBrain инициализирован")
        
        # Тестируем простой запрос
        test_query = "Привет!"
        print(f"Обработка запроса: '{test_query}'")
        
        start_time = time.time()
        try:
            result = brain.process_query(test_query)
            processing_time = time.time() - start_time
            
            print(f"Результат обработки: {result}")
            print(f"Время обработки: {processing_time:.4f} сек")
            
            if result:
                print("✅ Запрос обработан успешно")
                return True
            else:
                print("❌ Результат обработки пуст")
                return False
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Ошибка обработки запроса: {e}")
            print(f"Время до ошибки: {processing_time:.4f} сек")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Ошибка создания CoreBrain: {e}")
        traceback.print_exc()
        return False

def test_query_processor():
    """Тестирует QueryProcessor напрямую."""
    print("\n=== Тест QueryProcessor ===")
    
    try:
        # Создаем мок brain
        class MockBrain:
            def __init__(self):
                self.components = {}
                self.config = {}
                self.running = True
                # cache_dir нужен для корректной инициализации гибридного кэша в UTP
                self.cache_dir = os.path.join(os.getcwd(), 'cogniflex_cache')
        
        brain = MockBrain()
        
        from cogniflex.core.query_processor import QueryProcessor
        
        print("Создание QueryProcessor...")
        processor = QueryProcessor(brain)
        
        test_query = "Привет!"
        print(f"Обработка запроса: '{test_query}'")
        
        start_time = time.time()
        try:
            result = processor.process_query(test_query)
            processing_time = time.time() - start_time
            
            print(f"Результат: {result}")
            print(f"Время: {processing_time:.4f} сек")
            
            if result:
                print("✅ QueryProcessor работает")
                return True
            else:
                print("❌ QueryProcessor вернул пустой результат")
                return False
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Ошибка в QueryProcessor: {e}")
            print(f"Время до ошибки: {processing_time:.4f} сек")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Ошибка создания QueryProcessor: {e}")
        traceback.print_exc()
        return False

def test_response_generator():
    """Тестирует ResponseGenerator."""
    print("\n=== Тест ResponseGenerator ===")
    
    try:
        # Создаем мок brain
        class MockBrain:
            def __init__(self):
                self.components = {}
                self.config = {}
                self.running = True
        
        brain = MockBrain()
        
        from cogniflex.core.response_generator import ResponseGenerator
        
        print("Создание ResponseGenerator...")
        generator = ResponseGenerator(brain)
        
        test_query = "Привет!"
        print(f"Генерация ответа для: '{test_query}'")
        
        start_time = time.time()
        try:
            result = generator.generate_response(test_query)
            processing_time = time.time() - start_time
            
            print(f"Результат: {result}")
            print(f"Время: {processing_time:.4f} сек")
            
            if result:
                print("✅ ResponseGenerator работает")
                return True
            else:
                print("❌ ResponseGenerator вернул пустой результат")
                return False
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Ошибка в ResponseGenerator: {e}")
            print(f"Время до ошибки: {processing_time:.4f} сек")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Ошибка создания ResponseGenerator: {e}")
        traceback.print_exc()
        return False

def test_component_initialization():
    """Тестирует инициализацию компонентов."""
    print("\n=== Тест инициализации компонентов ===")
    
    try:
        from cogniflex.core.core_brain import CoreBrain
        
        brain = CoreBrain()
        # Сначала инициализируем, затем проверяем наличие компонентов
        print("\nИнициализация компонентов...")
        if brain.initialize():
            print("✅ Инициализация успешна")
            
            print("Проверка компонентов после инициализации:")
            components_to_check = [
                'config_manager', 'system_state_manager', 'resource_manager', 
                'system_metrics_manager', 'component_initializer', 'query_processor',
                'response_generator'
            ]
            missing_components = []
            for component in components_to_check:
                if hasattr(brain, component) and getattr(brain, component) is not None:
                    print(f"✅ {component}: присутствует")
                else:
                    print(f"❌ {component}: отсутствует")
                    missing_components.append(component)

            if missing_components:
                print(f"❌ Отсутствуют компоненты: {missing_components}")
                return False

            print("✅ Все компоненты присутствуют")

            # Проверяем статус компонентов
            status = brain.get_status()
            print(f"Статус системы: {status}")
            
            return True
        else:
            print("❌ Инициализация провалена")
            return False
        
    except Exception as e:
        print(f"❌ Ошибка тестирования компонентов: {e}")
        traceback.print_exc()
        return False

def test_ml_unit():
    """Тестирует MLUnit."""
    print("\n=== Тест MLUnit ===")
    
    try:
        # Создаем мок brain
        class MockBrain:
            def __init__(self):
                self.components = {}
                self.config = {}
                self.running = True
                self.response_generator = None
                # cache_dir нужен для корректной инициализации гибридного кэша в UTP
                self.cache_dir = os.path.join(os.getcwd(), 'cogniflex_cache')
        
        brain = MockBrain()
        
        from cogniflex.mlearning.ml_unit import MLUnit
        
        print("Создание MLUnit...")
        ml_unit = MLUnit(brain)
        
        test_text = "Привет!"
        print(f"Анализ текста: '{test_text}'")
        
        start_time = time.time()
        try:
            result = ml_unit.analyze_query(test_text)
            processing_time = time.time() - start_time
            
            print(f"Результат анализа: {result}")
            print(f"Время: {processing_time:.4f} сек")
            
            if result:
                print("✅ MLUnit работает")
                return True
            else:
                print("❌ MLUnit вернул пустой результат")
                return False
                
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Ошибка в MLUnit: {e}")
            print(f"Время до ошибки: {processing_time:.4f} сек")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Ошибка создания MLUnit: {e}")
        traceback.print_exc()
        return False

def analyze_error_patterns():
    """Анализирует паттерны ошибок."""
    print("\n=== Анализ паттернов ошибок ===")
    
    # Проверяем наличие основных файлов
    critical_files = [
        'cogniflex/core/core_brain.py',
        'cogniflex/core/query_processor.py', 
        'cogniflex/core/response_generator.py',
        'cogniflex/mlearning/ml_unit.py',
        'cogniflex/core/config_manager.py',
        'cogniflex/core/system_state.py',
        'cogniflex/core/resource_manager.py'
    ]
    
    missing_files = []
    for file_path in critical_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}: существует")
        else:
            print(f"❌ {file_path}: отсутствует")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Отсутствуют критические файлы: {missing_files}")
        return False
    
    print("✅ Все критические файлы присутствуют")
    return True

def main():
    """Основная функция диагностики."""
    print("🔍 ДИАГНОСТИКА ОБРАБОТКИ ЗАПРОСОВ COGNIFLEX")
    print("="*60)
    
    tests = [
        ("Анализ файлов", analyze_error_patterns),
        ("Инициализация компонентов", test_component_initialization),
        ("MLUnit", test_ml_unit),
        ("ResponseGenerator", test_response_generator),
        ("QueryProcessor", test_query_processor),
        ("CoreBrain Query", test_core_brain_query)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append(result)
            
            if result:
                print(f"✅ {test_name}: ПРОЙДЕН")
            else:
                print(f"❌ {test_name}: ПРОВАЛЕН")
                
        except Exception as e:
            print(f"❌ Критическая ошибка в {test_name}: {e}")
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
    print("="*60)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ ПРОЙДЕН" if results[i] else "❌ ПРОВАЛЕН"
        print(f"{test_name:25} : {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nОбщий результат: {passed}/{total} ({passed/total:.1%})")
    
    # Рекомендации
    print("\n🔧 РЕКОМЕНДАЦИИ:")
    
    if passed == total:
        print("🎉 Все тесты пройдены! Проблема может быть в GUI или конфигурации.")
    elif passed >= total * 0.8:
        print("⚠️ Большинство компонентов работает. Проверьте конкретные проблемные модули.")
    elif passed >= total * 0.5:
        print("🔧 Обнаружены серьезные проблемы. Требуется исправление ключевых компонентов.")
    else:
        print("❌ Критические ошибки. Система требует полной диагностики.")
    
    # Специфические рекомендации
    if not results[0]:  # analyze_error_patterns
        print("- Проверьте целостность файлов проекта")
    
    if not results[1]:  # test_component_initialization  
        print("- Проблемы с инициализацией компонентов")
    
    if not results[2]:  # test_ml_unit
        print("- MLUnit не работает корректно")
    
    if not results[3]:  # test_response_generator
        print("- ResponseGenerator имеет проблемы")
    
    if not results[4]:  # test_query_processor
        print("- QueryProcessor не обрабатывает запросы")
    
    if not results[5]:  # test_core_brain_query
        print("- Основная обработка запросов не работает")

if __name__ == "__main__":
    main()

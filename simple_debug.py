"""
Простая диагностика CogniFlex
"""

import os
import sys
import traceback

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def test_core_brain():
    print("=== Тест CoreBrain ===")
    
    try:
        print("Импорт CoreBrain...")
        from cogniflex.core.core_brain import CoreBrain
        print("✅ Импорт CoreBrain успешен")
        
        print("Создание CoreBrain...")
        brain = CoreBrain()
        print("✅ Создание CoreBrain успешно")
        
        print("Инициализация CoreBrain...")
        if brain.initialize():
            print("✅ Инициализация CoreBrain успешна")
            
            # Тест запроса
            print("Обработка тестового запроса...")
            result = brain.process_query("Привет!")
            if result:
                print(f"✅ Обработка запроса: {result}")
                return True
            else:
                print("❌ Пустой результат запроса")
                return False
        else:
            print("❌ Инициализация провалена")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка CoreBrain: {e}")
        traceback.print_exc()
        return False

def test_ml_unit():
    print("\n=== Тест MLUnit ===")
    
    try:
        class MockBrain:
            def __init__(self):
                self.components = {}
                self.config = {}
                self.running = True
                self.cache_dir = "."
        
        from cogniflex.mlearning.ml_unit import MLUnit
        print("✅ Импорт MLUnit успешен")
        
        brain = MockBrain()
        ml_unit = MLUnit(brain)
        print("✅ Создание MLUnit успешно")
        
        result = ml_unit.analyze_query("Привет!")
        if result:
            print(f"✅ Анализ запроса: {result}")
            return True
        else:
            print("❌ Пустой результат анализа")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка MLUnit: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔍 ПРОСТАЯ ДИАГНОСТИКА COGNIFLEX")
    print("="*50)
    
    results = []
    
    # Тест CoreBrain
    results.append(test_core_brain())
    
    # Тест MLUnit
    results.append(test_ml_unit())
    
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ:")
    print(f"CoreBrain: {'✅' if results[0] else '❌'}")
    print(f"MLUnit: {'✅' if results[1] else '❌'}")
    print(f"Общий результат: {sum(results)}/{len(results)}")

if __name__ == "__main__":
    main()

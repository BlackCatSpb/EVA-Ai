#!/usr/bin/env python3
"""
Быстрый тест исправлений ЕВА
"""

def test_core_brain():
    """Тестирует CoreBrain с новым методом process_query"""
    try:
        from eva_ai.core.core_brain import CoreBrain
        print("✅ Импорт CoreBrain успешен")
        
        brain = CoreBrain()
        print("✅ Создание CoreBrain успешно")
        
        if brain.initialize():
            print("✅ Инициализация CoreBrain успешна")
            
            # Тестируем новый метод process_query
            result = brain.process_query("Привет!")
            print(f"✅ process_query работает: {result.get('status', 'unknown')}")
            return True
        else:
            print("❌ Ошибка инициализации CoreBrain")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка CoreBrain: {e}")
        return False

def test_ml_unit():
    """Тестирует MLUnit с исправленным analyze_query"""
    try:
        from eva_ai.mlearning.ml_unit import MLUnit
        print("✅ Импорт MLUnit успешен")
        
        ml_unit = MLUnit()
        print("✅ Создание MLUnit успешно")
        
        # Тестируем исправленный analyze_query
        result = ml_unit.analyze_query("Тестовый запрос")
        print(f"✅ analyze_query работает: язык={result.get('language', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка MLUnit: {e}")
        return False

if __name__ == "__main__":
    print("🔧 ТЕСТ ИСПРАВЛЕНИЙ COGNIFLEX")
    print("=" * 40)
    
    results = []
    
    print("\n=== Тест CoreBrain ===")
    results.append(test_core_brain())
    
    print("\n=== Тест MLUnit ===") 
    results.append(test_ml_unit())
    
    print("\n" + "=" * 40)
    print(f"📊 РЕЗУЛЬТАТЫ: {sum(results)}/{len(results)} тестов пройдено")
    
    if all(results):
        print("🎉 Все исправления работают!")
    else:
        print("⚠️ Некоторые проблемы остаются")

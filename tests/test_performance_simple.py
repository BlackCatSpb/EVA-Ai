"""
Упрощенный тест производительности для диагностики
"""

import os
import sys
import time

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def test_basic_imports():
    """Тестирует базовые импорты."""
    print("=== Тест импортов ===")
    
    try:
        print("Импорт UnifiedTextProcessor...")
        from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
        print("✅ UnifiedTextProcessor импортирован")
        
        print("Импорт HybridTokenCache...")
        from eva_ai.memory.hybrid_token_cache import HybridTokenCache
        print("✅ HybridTokenCache импортирован")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Тестирует базовую функциональность."""
    print("\n=== Тест базовой функциональности ===")
    
    try:
        # Создаем минимальный brain
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        print("✅ MockBrain создан")
        
        # Создаем процессор
        from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
        processor = UnifiedTextProcessor(brain=brain, use_async=False, max_workers=1)
        print("✅ UnifiedTextProcessor создан")
        
        # Тестируем простую обработку
        test_text = "Это простой тест обработки текста."
        print(f"Обработка текста: '{test_text}'")
        
        start_time = time.time()
        result = processor.process_text(test_text)
        processing_time = time.time() - start_time
        
        if result:
            print(f"✅ Обработка завершена за {processing_time:.4f}сек")
            if 'keywords' in result:
                print(f"   Найдено ключевых слов: {len(result['keywords'])}")
                print(f"   Ключевые слова: {result['keywords'][:5]}")  # Первые 5
            return True
        else:
            print("❌ Результат обработки пуст")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка в тесте функциональности: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_creation():
    """Тестирует создание кэша."""
    print("\n=== Тест создания кэша ===")
    
    try:
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        
        from eva_ai.memory.hybrid_token_cache import HybridTokenCache
        cache = HybridTokenCache(brain, max_memory_tokens=100)
        print("✅ HybridTokenCache создан")
        
        # Тестируем базовые операции кэша
        test_key = "test_key"
        test_tokens = ["тест", "токен", "кэш"]
        
        print(f"Добавление в кэш: {test_key} -> {test_tokens}")
        cache.add_token(test_key, test_tokens)
        print("✅ Токены добавлены в кэш")
        
        print(f"Получение из кэша: {test_key}")
        cached_tokens = cache.get_token(test_key)
        
        if cached_tokens:
            print(f"✅ Токены получены из кэша: {cached_tokens}")
            return True
        else:
            print("❌ Токены не найдены в кэше")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка в тесте кэша: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Тестирует сравнение производительности."""
    print("\n=== Тест сравнения производительности ===")
    
    try:
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        
        from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
        from eva_ai.memory.hybrid_token_cache import HybridTokenCache
        
        # Тестовые тексты
        test_texts = [
            "Машинное обучение использует алгоритмы для анализа данных.",
            "Искусственный интеллект революционизирует современные технологии.",
            "Нейронные сети позволяют решать сложные задачи классификации."
        ]
        
        # Тест без кэша
        print("Тест без кэша:")
        processor_no_cache = UnifiedTextProcessor(brain=brain, use_async=False)
        
        times_no_cache = []
        for i, text in enumerate(test_texts):
            start_time = time.time()
            result = processor_no_cache.process_text(text)
            processing_time = time.time() - start_time
            times_no_cache.append(processing_time)
            print(f"  Текст {i+1}: {processing_time:.4f}сек")
        
        avg_time_no_cache = sum(times_no_cache) / len(times_no_cache)
        print(f"Среднее время без кэша: {avg_time_no_cache:.4f}сек")
        
        # Тест с кэшем
        print("\nТест с кэшем:")
        cache = HybridTokenCache(brain, max_memory_tokens=500)
        processor_with_cache = UnifiedTextProcessor(brain=brain, use_async=False, hybrid_cache=cache)
        
        times_with_cache = []
        for i, text in enumerate(test_texts):
            start_time = time.time()
            result = processor_with_cache.process_text(text)
            processing_time = time.time() - start_time
            times_with_cache.append(processing_time)
            print(f"  Текст {i+1}: {processing_time:.4f}сек")
        
        avg_time_with_cache = sum(times_with_cache) / len(times_with_cache)
        print(f"Среднее время с кэшем: {avg_time_with_cache:.4f}сек")
        
        # Повторная обработка для проверки кэширования
        print("\nПовторная обработка (проверка кэша):")
        times_cached = []
        for i, text in enumerate(test_texts):
            start_time = time.time()
            result = processor_with_cache.process_text(text)
            processing_time = time.time() - start_time
            times_cached.append(processing_time)
            status = "из кэша" if processing_time < 0.001 else "обработан"
            print(f"  Текст {i+1}: {processing_time:.6f}сек ({status})")
        
        avg_time_cached = sum(times_cached) / len(times_cached)
        print(f"Среднее время повторной обработки: {avg_time_cached:.6f}сек")
        
        # Анализ результатов
        if avg_time_no_cache > 0:
            speedup_cache = avg_time_no_cache / max(0.000001, avg_time_with_cache)
            speedup_cached = avg_time_no_cache / max(0.000001, avg_time_cached)
            
            print(f"\n📊 Результаты:")
            print(f"Ускорение с кэшем: {speedup_cache:.2f}x")
            print(f"Ускорение при повторном использовании: {speedup_cached:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в тесте производительности: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_context_quality():
    """Анализирует качество понимания контекста."""
    print("\n=== Анализ качества понимания контекста ===")
    
    try:
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        
        from eva_ai.mlearning.unified_text_processor import UnifiedTextProcessor
        processor = UnifiedTextProcessor(brain=brain)
        
        # Тестовые контексты разной сложности
        test_contexts = [
            {
                'name': 'Простой',
                'text': 'Машинное обучение использует алгоритмы.',
                'expected_concepts': ['машинное', 'обучение', 'алгоритмы']
            },
            {
                'name': 'Средний',
                'text': 'Искусственный интеллект и нейронные сети революционизируют технологии обработки данных.',
                'expected_concepts': ['интеллект', 'нейронные', 'сети', 'технологии', 'данных']
            },
            {
                'name': 'Сложный',
                'text': 'Трансформеры с механизмом внимания позволяют моделям BERT и GPT понимать контекстуальные зависимости в естественном языке.',
                'expected_concepts': ['трансформеры', 'внимание', 'bert', 'gpt', 'контекстуальные', 'языке']
            }
        ]
        
        quality_scores = []
        
        for context in test_contexts:
            print(f"\nАнализ контекста '{context['name']}':")
            print(f"Текст: {context['text']}")
            
            result = processor.process_text(context['text'])
            
            if result and 'keywords' in result:
                # Обрабатываем различные форматы ключевых слов
                raw_keywords = result['keywords']
                if isinstance(raw_keywords, list) and raw_keywords:
                    if isinstance(raw_keywords[0], tuple):
                        # Формат [(слово, вес), ...]
                        keywords = [kw[0].lower() for kw in raw_keywords]
                    else:
                        # Формат [слово, ...]
                        keywords = [str(kw).lower() for kw in raw_keywords]
                else:
                    keywords = []
                
                expected = [exp.lower() for exp in context['expected_concepts']]
                
                # Подсчитываем совпадения
                matches = sum(1 for exp in expected if any(exp in kw for kw in keywords))
                quality = matches / len(expected) if expected else 0
                
                quality_scores.append(quality)
                
                print(f"  Извлеченные ключевые слова: {keywords[:10]}")  # Первые 10
                print(f"  Совпадений с ожидаемыми: {matches}/{len(expected)}")
                print(f"  Качество понимания: {quality:.2%}")
                
                # Анализ глубины (разнообразие токенов)
                unique_ratio = len(set(keywords)) / max(1, len(keywords))
                print(f"  Глубина анализа (уникальность): {unique_ratio:.2%}")
                
                # Анализ ширины (покрытие текста)
                text_words = context['text'].lower().split()
                coverage = sum(1 for kw in keywords if any(kw in word for word in text_words))
                coverage_ratio = coverage / max(1, len(text_words))
                print(f"  Ширина анализа (покрытие): {coverage_ratio:.2%}")
            else:
                print("  ❌ Не удалось извлечь ключевые слова")
                quality_scores.append(0)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        print(f"\n📊 Средняя качество понимания контекста: {avg_quality:.2%}")
        
        return avg_quality > 0.3  # Минимум 30% качества
        
    except Exception as e:
        print(f"❌ Ошибка в анализе качества: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 УПРОЩЕННЫЙ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)
    
    tests = [
        ("Импорты", test_basic_imports),
        ("Базовая функциональность", test_basic_functionality),
        ("Создание кэша", test_cache_creation),
        ("Сравнение производительности", test_performance_comparison),
        ("Качество понимания контекста", analyze_context_quality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Критическая ошибка в {test_name}: {e}")
            results[test_name] = False
        
        if not results[test_name]:
            print(f"⚠️ Тест '{test_name}' провален, остальные тесты могут быть неточными")
    
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ УПРОЩЕННОГО ТЕСТА")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{test_name:30} : {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nОбщий результат: {passed}/{total} тестов пройдено ({passed/total:.1%})")
    
    if passed >= total * 0.8:
        print("🎉 Основная функциональность работает корректно!")
    else:
        print("⚠️ Обнаружены проблемы, требующие исправления.")

"""
Тест асинхронной токенизации для CogniFlex
"""

import os
import sys
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Добавляем путь к проекту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cogniflex'))

def test_unified_text_processor():
    """Тестирует UnifiedTextProcessor с асинхронной обработкой."""
    print("=== Тест UnifiedTextProcessor ===")
    
    try:
        # Создаем минимальный brain
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        
        # Импортируем и создаем процессор
        from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
        
        print("Создание UnifiedTextProcessor...")
        processor = UnifiedTextProcessor(brain=brain, use_async=True, max_workers=2)
        print("✅ UnifiedTextProcessor создан")
        
        # Тест базовой обработки текста
        test_texts = [
            "Это простой тест обработки текста.",
            "CogniFlex использует машинное обучение для анализа.",
            "Асинхронная обработка повышает производительность."
        ]
        
        results = []
        for i, text in enumerate(test_texts):
            print(f"\nОбработка текста {i+1}: '{text[:30]}...'")
            
            start_time = time.time()
            result = processor.process_text(text)
            processing_time = time.time() - start_time
            
            if result:
                print(f"✅ Обработано за {processing_time:.4f}сек")
                print(f"   Ключевые слова: {len(result.get('keywords', []))}")
                print(f"   Токены: {len(result.get('tokens', []))}")
                results.append(result)
            else:
                print("❌ Ошибка обработки")
        
        print(f"\n✅ Обработано {len(results)}/{len(test_texts)} текстов")
        return len(results) == len(test_texts)
        
    except Exception as e:
        print(f"❌ Ошибка в тесте UnifiedTextProcessor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_callback_processing():
    """Тестирует асинхронную обработку с callback."""
    print("\n=== Тест асинхронной обработки с callback ===")
    
    try:
        # Создаем минимальный brain
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        
        from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
        processor = UnifiedTextProcessor(brain=brain, use_async=True, max_workers=4)
        
        # Результаты callback'ов
        callback_results = []
        completed_count = 0
        
        def test_callback(tokens, context, error=None):
            nonlocal completed_count
            if error:
                print(f"❌ Callback ошибка: {error}")
            else:
                callback_results.append({
                    "context": context,
                    "token_count": len(tokens) if tokens else 0
                })
                print(f"✅ Callback: обработано {len(tokens) if tokens else 0} токенов")
            completed_count += 1
        
        # Тестовые тексты для асинхронной обработки
        async_texts = [
            "Первый текст для асинхронной обработки",
            "Второй текст с более сложной структурой и дополнительными словами",
            "Третий текст для проверки параллельной обработки нескольких запросов",
            "Четвертый текст для тестирования производительности системы"
        ]
        
        # Запускаем асинхронную обработку
        start_time = time.time()
        
        for i, text in enumerate(async_texts):
            context = {"text_id": i, "original_text": text}
            
            # Проверяем, есть ли метод tokenize для асинхронной обработки
            if hasattr(processor, 'tokenize'):
                processor.tokenize(text, test_callback, context, priority=5)
            else:
                # Fallback на синхронную обработку
                try:
                    result = processor.process_text(text)
                    tokens = result.get('keywords', []) if result else []
                    test_callback(tokens, context)
                except Exception as e:
                    test_callback(None, context, str(e))
        
        # Ждем завершения всех операций
        timeout = 15  # 15 секунд максимум
        while completed_count < len(async_texts) and timeout > 0:
            time.sleep(0.1)
            timeout -= 0.1
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Асинхронная обработка завершена:")
        print(f"   Обработано: {completed_count}/{len(async_texts)} текстов")
        print(f"   Время: {processing_time:.4f}сек")
        print(f"   Скорость: {len(async_texts)/processing_time:.1f} текстов/сек")
        
        return completed_count == len(async_texts)
        
    except Exception as e:
        print(f"❌ Ошибка в тесте асинхронной обработки: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_concurrent_processing():
    """Тестирует параллельную обработку текстов."""
    print("\n=== Тест параллельной обработки ===")
    
    try:
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        
        from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
        processor = UnifiedTextProcessor(brain=brain, use_async=True, max_workers=3)
        
        # Создаем большие тексты для тестирования
        large_texts = []
        for i in range(6):
            text = f"Большой текст номер {i+1}. " * 50  # Повторяем 50 раз
            large_texts.append(text)
        
        # Тестируем последовательную обработку
        print("Последовательная обработка...")
        sequential_start = time.time()
        sequential_results = []
        
        for text in large_texts[:3]:  # Берем только 3 текста для сравнения
            result = processor.process_text(text)
            if result:
                sequential_results.append(result)
        
        sequential_time = time.time() - sequential_start
        
        # Тестируем параллельную обработку
        print("Параллельная обработка...")
        parallel_start = time.time()
        parallel_results = []
        
        def process_text_worker(text):
            return processor.process_text(text)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_text_worker, text) for text in large_texts[:3]]
            
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    if result:
                        parallel_results.append(result)
                except Exception as e:
                    print(f"Ошибка в параллельной обработке: {e}")
        
        parallel_time = time.time() - parallel_start
        
        print(f"\n✅ Результаты сравнения:")
        print(f"   Последовательно: {len(sequential_results)} текстов за {sequential_time:.4f}сек")
        print(f"   Параллельно: {len(parallel_results)} текстов за {parallel_time:.4f}сек")
        
        if parallel_time < sequential_time:
            speedup = sequential_time / parallel_time
            print(f"   Ускорение: {speedup:.2f}x")
        
        return len(parallel_results) >= len(sequential_results)
        
    except Exception as e:
        print(f"❌ Ошибка в тесте параллельной обработки: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenization_quality():
    """Тестирует качество токенизации."""
    print("\n=== Тест качества токенизации ===")
    
    try:
        class MockBrain:
            def __init__(self):
                self.cache_dir = os.path.join(os.path.dirname(__file__), "test_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self.components = {}
        
        brain = MockBrain()
        
        from cogniflex.mlearning.unified_text_processor import UnifiedTextProcessor
        processor = UnifiedTextProcessor(brain=brain)
        
        # Тестовые тексты с известными ключевыми словами
        test_cases = [
            {
                "text": "Машинное обучение и искусственный интеллект революционизируют технологии",
                "expected_keywords": ["машинное", "обучение", "искусственный", "интеллект", "технологии"]
            },
            {
                "text": "Python является популярным языком программирования для анализа данных",
                "expected_keywords": ["python", "язык", "программирование", "анализ", "данные"]
            },
            {
                "text": "Нейронные сети используются для решения сложных задач классификации",
                "expected_keywords": ["нейронные", "сети", "задач", "классификации"]
            }
        ]
        
        quality_scores = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\nТест {i+1}: '{test_case['text'][:40]}...'")
            
            result = processor.process_text(test_case["text"])
            
            if result and 'keywords' in result:
                found_keywords = [kw.lower() for kw in result['keywords']]
                expected = [kw.lower() for kw in test_case['expected_keywords']]
                
                # Подсчитываем совпадения
                matches = sum(1 for exp in expected if any(exp in found for found in found_keywords))
                quality = matches / len(expected) if expected else 0
                
                quality_scores.append(quality)
                
                print(f"   Найдено ключевых слов: {len(found_keywords)}")
                print(f"   Совпадений с ожидаемыми: {matches}/{len(expected)}")
                print(f"   Качество: {quality:.2%}")
            else:
                print("   ❌ Не удалось извлечь ключевые слова")
                quality_scores.append(0)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        print(f"\n✅ Средняя качество токенизации: {avg_quality:.2%}")
        
        return avg_quality > 0.3  # Минимум 30% качества
        
    except Exception as e:
        print(f"❌ Ошибка в тесте качества: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов асинхронной токенизации")
    print("=" * 60)
    
    tests = [
        ("UnifiedTextProcessor", test_unified_text_processor),
        ("Асинхронная обработка", test_async_callback_processing),
        ("Параллельная обработка", test_concurrent_processing),
        ("Качество токенизации", test_tokenization_quality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Критическая ошибка в {test_name}: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{test_name:25} : {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nОбщий результат: {passed}/{total} тестов пройдено ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 Все тесты асинхронной токенизации успешно пройдены!")
    else:
        print("⚠️  Некоторые тесты провалены. Требуется доработка.")
